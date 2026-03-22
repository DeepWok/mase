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
