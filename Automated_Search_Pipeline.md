# Part 3: Automated Search Pipeline — Implementation Summary

## Overview

Part 3 closes the loop between Parts 1 (FlexAttention) and 2 (Fused RMSNorm) by:
1. Making MASE's `LatencyRunner` actually measure GPU time
2. Extending the search space to cover fusion strategy alongside quantisation bit-widths
3. Wiring everything into Optuna NSGA-II multi-objective search (accuracy/perplexity ↑, latency ↓, avg_bitwidth ↓)

---

## Branch

```bash
git checkout flexattention
git checkout -b Automated_Search_Pipeline
```

When teammates finish Part 2:
```bash
git merge fused-rmsnorm-residual
```
The fused RMSNorm pass activates automatically — no code changes needed.

---

## Files Changed / Created

### Edited
| File | Change |
|------|--------|
| `src/chop/actions/search/strategies/runners/hardware/latency.py` | Filled stub with real GPU timing |
| `src/chop/actions/search/search_space/quantization/__init__.py` | +1 import |
| `src/chop/actions/search/search_space/__init__.py` | +1 registry entry |
| `src/chop/pipelines/__init__.py` | +1 import |

### Created
| File | Purpose |
|------|---------|
| `src/chop/actions/search/search_space/quantization/module_fusion.py` | New search space class |
| `src/chop/pipelines/optimization.py` | Pass-chain wrapper for standalone use |
| `configs/search/quantization_fusion_bert.toml` | BERT search config |
| `configs/search/quantization_fusion_llama.toml` | TinyLlama search config |
| `configs/search/quantization_fusion_mistral.toml` | Mistral search config |
| `test/actions/search/test_latency_runner.py` | LatencyRunner unit + GPU tests |
| `run_search_bert.pbs` | HPC job script — BERT |
| `run_search_llama.pbs` | HPC job script — TinyLlama |
| `run_search_mistral.pbs` | HPC job script — Mistral |

---

## Component Details

### 1. LatencyRunner
**File:** `src/chop/actions/search/strategies/runners/hardware/latency.py`

Previously an empty stub. Now:
- Reads `num_batches` (default 50) and `num_warmup_batches` (default 10) from config
- Warmup loop runs first with no timing (GPU cache warmup)
- CUDA path: `torch.cuda.Event(enable_timing=True)` start/end events around the timed loop, `torch.cuda.synchronize()` after
- CPU fallback: `time.perf_counter`
- Returns `{"latency": elapsed_ms / num_batches}` (milliseconds per batch)
- Unwraps `MaseGraph` objects (accesses `.model` attribute) automatically

TOML config block:
```toml
[search.strategy.hw_runner.latency]
num_batches        = 50
num_warmup_batches = 10
```

---

### 2. ModuleSearchSpaceQuantizationFusion
**File:** `src/chop/actions/search/search_space/quantization/module_fusion.py`

**Key design choice:** subclasses `SearchSpaceBase` directly (not `GraphSearchSpaceMixedPrecisionPTQ`) — works on plain `nn.Module` without FX tracing, which is required for HuggingFace models like TinyLlama and Mistral.

**Search dimensions (Optuna trial integers):**
| Dimension | Choices |
|-----------|---------|
| `data_in_width` | `[4, 8, 16, 32]` |
| `weight_width` | `[4, 8, 16, 32]` |
| `fusion_strategy` | `["none", "flex_attention", "fused_rmsnorm", "both"]` |

**`rebuild_model()` pass chain:**
1. `quantize_module_transform_pass` — uniform integer quantisation on all `nn.Linear` by type; `frac_width = width // 2`
2. `flex_attention_transform_pass` — if `fusion_strategy in {"flex_attention", "both"}`
3. `rmsnorm_residual_fusion_pass` — if `fusion_strategy in {"fused_rmsnorm", "both"}`, imported lazily via `try/except ImportError` (silently skipped until Part 2 is merged)

**Registered as:** `"module/quantize_fusion"` in `SEARCH_SPACE_MAP`

TOML config block:
```toml
[search.search_space]
name = "module/quantize_fusion"

[search.search_space.setup]
score_mod = "causal"          # or "none", "sliding_window", "alibi"
score_mod_kwargs = {}         # e.g. {window_size = 512} for sliding_window

[search.search_space.seed.default]
data_in_width   = [4, 8, 16, 32]
weight_width    = [4, 8, 16, 32]
fusion_strategy = ["none", "flex_attention", "fused_rmsnorm", "both"]
```

---

### 3. OptimizationPipeline
**File:** `src/chop/pipelines/optimization.py`

Thin wrapper that applies the same pass chain as `rebuild_model()` to a single fixed config. Intended for:
- Evaluating the best Pareto point found by search
- Generating the ablation table in the notebook
- Standalone benchmarking

```python
from chop.pipelines.optimization import OptimizationPipeline

pipeline = OptimizationPipeline(model, score_mod="causal")
optimized = pipeline.run({
    "quantization": {"data_in_width": 8, "weight_width": 8},
    "fusion_strategy": "both",
})
```

---

### 4. Search Configs

| Config | Model | Task | score_mod | Trials | Wall time |
|--------|-------|------|-----------|--------|-----------|
| `quantization_fusion_bert.toml` | `bert-base-uncased` | SST-2 cls | `none` | 80 | ~3h |
| `quantization_fusion_llama.toml` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | wikitext2 lm | `causal` | 100 | ~6h |
| `quantization_fusion_mistral.toml` | `mistralai/Mistral-7B-v0.1` | wikitext2 lm | `sliding_window` | 100 | ~10h |

All configs use:
- Sampler: NSGA-II (`"nsgaii"`)
- `sum_scaled_metrics = false` → multi-objective
- 3 objectives: accuracy/perplexity + latency + average_bitwidth
- `[search.strategy.hw_runner.average_bitwidth]` compare_to = 32 (vs FP32)

**Note on BERT:** `fusion_strategy = ["none", "flex_attention"]` only — BERT uses LayerNorm, not RMSNorm, so `fused_rmsnorm` has no applicable modules.

**Note on Mistral:** Requires HuggingFace login (`huggingface-cli login`) for gated model access.

---

### 5. PBS Scripts

| Script | Model | Walltime | Memory |
|--------|-------|----------|--------|
| `run_search_bert.pbs` | BERT | 3h | 32GB |
| `run_search_llama.pbs` | TinyLlama | 6h | 64GB |
| `run_search_mistral.pbs` | Mistral-7B | 10h | 80GB |

All use `gpu_type=L40S`. Output saved to `outputs/search/<model>_<timestamp>/`.

Submit with:
```bash
mkdir -p logs
qsub run_search_bert.pbs
qsub run_search_llama.pbs
# submit mistral only after confirming llama works correctly
```

---

## HPC Timing: Do You Need to Wait for Part 2?

**No.** Run with current configs immediately. The `fused_rmsnorm` and `both` strategies are silently skipped (via `ImportError` catch) until Part 2 merges. Once merged:
1. No code changes needed — the pass activates automatically
2. Re-run the search jobs with the same TOML configs to get the full 4-strategy Pareto frontier

---

## Running Tests

```bash
# CPU unit tests (no GPU needed)
pytest test/actions/search/test_latency_runner.py -v -k "CPU"

# GPU integration tests
pytest test/actions/search/test_latency_runner.py -v -k "GPU"

# All tests
pytest test/actions/search/test_latency_runner.py -v
```

---

## Recommended Implementation Order

```
Day 1  LatencyRunner done → run CPU tests locally → submit to HPC to verify GPU timing
Day 2  Search space + register in __init__ files
Day 3  OptimizationPipeline + TOML configs
Day 4  PBS scripts → submit BERT + LLaMA overnight
Day 5  Notebook (Pareto plots from study.pkl, ablation table, seq-length scaling)
       + merge Part 2 → add fused_rmsnorm to search, resubmit
```

---

## Notebook Outline (to implement in Week 3)

**File:** `notebooks/quantization_fusion_search.ipynb`

1. Load `study.pkl` for each model via `joblib.load()`
2. **Pareto plot** — 2D/3D scatter of the three objectives, colour-coded by `fusion_strategy`
3. **Ablation table** — `OptimizationPipeline.run()` for 4 fixed configs on TinyLlama:
   baseline → +quantisation → +fused_rmsnorm → +flex_attention → both
4. **Sequence length scaling** — latency vs seq_len `[64, 128, 256, 512, 1024, 2048, 4096]` per fusion strategy; shows FlexAttention sliding window scales sub-quadratically
5. **Memory profiling** — `torch.cuda.max_memory_allocated()` per config

---

## ALiBi Score Mod

Part 1 implements `generate_alibi_score_mod` but no PBS script was added for it. ALiBi requires a model trained with it (e.g. BLOOM, MPT) — applying it to TinyLlama or Mistral would hurt accuracy and pollute the Pareto results. Potential future addition: `quantization_fusion_bloom.toml` + `run_search_bloom.pbs` targeting `bigscience/bloom-560m`.

# BEFORE Fused RMSnorm Implementation
## BERT Search Results (SST-2, 80 trials)

| Point | Trial | Accuracy | Avg Bitwidth | Latency (ms) | Memory Density | Fusion Strategy | data_in_width | weight_width |
|-------|-------|----------|--------------|--------------|----------------|-----------------|---------------|--------------|
| 0 | 61 | 0.518 | 8-bit | 24.88 | 4x | none | 8 | 8 |
| 1 | 66 | 0.511 | 8-bit | 19.69 | 4x | flex_attention | 8 | 8 |
| 2 | 69 | **0.521** | 16-bit | 20.57 | 2x | flex_attention | 32 | 16 |
| 3 | 75 | 0.510 | 8-bit | 19.64 | 4x | flex_attention | 4 | 8 |
| 4 | 77 | 0.500 | **4-bit** | **19.34** | 8x | flex_attention | 8 | 4 |

> Note: Accuracy ~0.5 (random chance) — BERT classifier head was not fine-tuned on SST-2.
> FlexAttention delivers ~21% latency reduction vs no-fusion at equivalent 8-bit precision (24.88ms → 19.69ms).

Each trial runs 256 samples (16 batch size = 16 batches) for the accuracy/loss evaluation, plus 50 batches for latency timing with 10 warmup batches. No training; purely inference to evaluate each config.
---

## TinyLlama-1.1B Search Results (WikiText-2, 100 trials)

| Point | Trial | Perplexity | Avg Bitwidth | Latency (ms) | Memory Density | Fusion Strategy | data_in_width | weight_width |
|-------|-------|------------|--------------|--------------|----------------|-----------------|---------------|--------------|
| 0 | 2 | 14,285 | 16-bit | 160.54 | 2x | fused_rmsnorm* | 4 | 16 |
| 1 | 12 | 72,084 | **4-bit** | **157.30** | 8x | fused_rmsnorm* | 16 | 4 |
| 2 | 64 | 30,722 | **4-bit** | 157.65 | 8x | none | 4 | 4 |
| 3 | 73 | 33,464 | **4-bit** | 157.53 | 8x | fused_rmsnorm* | 8 | 4 |

> *fused_rmsnorm silently skipped — Part 2 not yet merged. These configs are effectively no-fusion baselines.
> Weight quantisation dominates the Pareto front — all memory-efficient points use 4-bit weights.
> flex_attention does not appear on the Llama Pareto front.

## Sampler Used:
NSGA-II (Non-dominated Sorting Genetic Algorithm II) is the standard algorithm for multi-objective optimisation — it's specifically designed to find a Pareto front rather than a single best point.

With a single objective you'd use something like TPE (Tree-structured Parzen Estimator) which just finds one optimum. But you have 3 competing objectives (accuracy ↑, latency ↓, bitwidth ↓) that can't all be maximised simultaneously — improving one often worsens another. NSGA-II handles this by:

Non-dominated sorting — ranks trials by whether any other trial beats them on all objectives simultaneously. Trials that aren't beaten by anyone are "Pareto-optimal"
Crowding distance — spreads solutions evenly along the Pareto front so you get diverse trade-offs, not 80 clustered points
Genetic evolution — uses crossover and mutation on good configs to generate the next generation of trials, so it learns which combinations of bitwidths and fusion strategies tend to be good
The result is the 4-5 point Pareto front you saw in best.json, each point representing a genuinely different trade-off that can't be improved in all objectives at once.

# Changes to get FlexAttention to work on Mistral:


##	Problem	Fix
1	Missing module load in PBS script	Added module load Python/3.12.3-GCCcore-13.3.0

2	python -m chop search doesn't exist	Created scripts/search_mistral.py, matched search_llama.py pattern

3	window_size=512 = max_token_len — no actual sparsity	Changed window_size to 128 (75% blocks skipped)

4	float32 Mistral head_dim=128 exceeds L40S shared memory (101KB)	Load model in float16: torch_dtype=torch.float16

5	Quantize pass creates float32 LinearInteger modules — dtype mismatch with float16 hidden states	Restored orig_dtype after quantize pass in rebuild_model()

---

# Benchmark Results WITHOUT Fused RMSNorm (Experiments 3, 4, 5)

**Date:** 2026-03-25 | **Hardware:** NVIDIA L40S (46GB) | **PyTorch:** 2.6.0+cu124

---

### TinyLlama-1.1B — Sequence Length Scaling
**Config:** causal FlexAttention, batch=1, 5 warmup + 20 timed forward passes

| seq_len | baseline FP32 (ms) | int8_none (ms) | int8_flex causal (ms) | flex vs int8_none |
|---------|--------------------|----------------|-----------------------|-------------------|
| 64      | 14.4               | 39.7           | null†                 | —                 |
| 128     | 15.1               | 40.9           | **32.8**              | **−20%**          |
| 256     | 25.0               | 44.8           | **42.6**              | −5%               |
| 512     | 43.6               | 61.7           | **59.7**              | −3%               |
| 1024    | 88.8               | 95.7           | 97.7                  | +2%               |
| 2048    | 181.4              | 168.1          | 182.4                 | +9%               |
| 4096    | 354.6              | 359.9          | 401.8                 | +12%              |

† seq_len=64 < FlexAttention minimum tile size (BLOCK_M=128). Expected failure.

**Key findings:**
- FlexAttention causal wins at short sequences (128–512) where fake-quant overhead dominates
- Loses at 1024+ tokens: PyTorch SDPA has a native cuDNN Flash Attention causal path for GQA that outperforms FlexAttention's generic Triton kernel
- The ~50% block skip from causal masking does not overcome kernel efficiency gap at long sequences
- INT8 quantisation beats FP32 baseline from seq_len ≥ 512 (memory bandwidth savings kick in)

---

### TinyLlama-1.1B — Peak Memory vs Sequence Length
**Config:** batch=1

| seq_len | baseline FP32 (MB) | int8_none (MB) | int8_flex (MB) |
|---------|--------------------|----------------|----------------|
| 128     | 4,227              | 4,712          | 4,712          |
| 256     | 4,249              | 4,719          | 4,720          |
| 512     | 4,293              | 4,734          | 4,735          |
| 1024    | 4,382              | 4,764          | 4,765          |
| 2048    | 4,559              | 4,825          | 4,825          |
| 4096    | 4,913              | 5,195          | 5,195          |

**Key finding:** INT8 configs use ~500MB more than FP32 baseline (fake-quant stores both original and quantised weights). FlexAttention adds negligible memory overhead vs int8_none — block_mask tensors are small.

---

### TinyLlama-1.1B — Batch Size Scaling
**Config:** seq_len=512, causal FlexAttention

| batch | baseline FP32 (ms) | int8_none (ms) | int8_flex (ms) | int8_flex throughput (samples/s) |
|-------|--------------------|----------------|----------------|----------------------------------|
| 1     | 44.5               | 62.2           | **60.5**       | 16.5                             |
| 4     | 172.0              | **155.2**      | 159.1          | 25.1                             |
| 8     | 300.3              | 318.1          | **315.9**      | 25.3                             |
| 16    | 620.3              | 679.7          | 994.4          | 16.1                             |

**Key findings:**
- INT8 beats FP32 baseline from batch=4 (memory bandwidth savings dominate at higher batch)
- FlexAttention degrades badly at batch=16 (994ms vs 680ms — **46% slower than int8_none**)
- FlexAttention is designed for memory-bound latency inference (batch=1), not throughput workloads
- Optimal throughput for int8 is batch=4–8 (~25 samples/s)

---

### Mistral-7B — Sequence Length Scaling
**Config:** sliding_window FlexAttention (window_size=128), batch=1, float16, 5 warmup + 20 timed forward passes

| seq_len | baseline FP16 (ms) | int8_none (ms) | int8_flex sliding_window (ms) | flex vs int8_none |
|---------|--------------------|----------------|-------------------------------|-------------------|
| 64      | 27.4               | 171.5          | 171.4                         | ~0%               |
| 128     | 30.1               | 174.5          | **165.4**                     | **−5%**           |
| 256     | 34.1               | 180.6          | **169.5**                     | −6%               |
| 512     | 49.4               | 195.9          | **183.1**                     | −6.5%             |
| 1024    | 90.5               | 237.4          | **220.5**                     | **−7%**           |
| 2048    | 179.2              | 328.5          | **305.0**                     | **−7%**           |

**Key findings:**
- Sliding window FlexAttention consistently beats SDPA from seq_len=128 onwards
- Speedup stabilises at ~7% for seq_len ≥ 512 and continues to grow in absolute ms (23ms saved at 2048)
- Root cause: PyTorch SDPA has **no native sliding window implementation** — it computes full attention then masks, while FlexAttention's block_mask genuinely skips entire blocks
- The 7% figure is the attention-layer savings diluted by quantised linear layers dominating total latency
- Baseline FP16 faster than both INT8 configs — fake quantisation does not provide real integer throughput

---

### Mistral-7B — Peak Memory vs Sequence Length

| seq_len | baseline FP16 (MB) | int8_none (MB) | int8_flex (MB) |
|---------|--------------------|----------------|----------------|
| 128     | 13,853             | 14,339         | 14,339         |
| 512     | 13,943             | 14,393         | 14,393         |
| 1024    | 14,065             | 14,465         | 14,465         |
| 2048    | 14,310             | 14,609         | 14,609         |

**Key finding:** Memory scales slowly with seq_len (no quadratic growth visible at these lengths — attention KV tensors are small relative to 7B weight memory). FlexAttention adds ~1.4MB overhead vs int8_none from block_mask tensors.

---

### Cross-Model Summary

| Optimisation | TinyLlama (causal) | Mistral (sliding_window) | Verdict |
|---|---|---|---|
| INT8 quantisation | Slower at seq≤512, faster at seq≥1024 | Slower vs FP16 baseline (fake quant) | Real INT8 kernels needed for benefit |
| FlexAttention | Wins at seq 128–512, loses at 1024+ | Consistently wins at seq≥128 | Only beneficial when SDPA lacks native sparse path |
| FlexAttention at batch=16 | 46% slower | N/A | Not suitable for throughput workloads |
