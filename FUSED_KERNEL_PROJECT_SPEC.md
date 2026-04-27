# Part 2: Fused Add+RMSNorm Triton Kernel — Complete Project Spec

---

## 1. Project Overview

**Part:** 2 of 3 in the ADLS kernel-fusion-aware optimisation pipeline for MASE  
**Author:** Dorijan Donaj Magašić (dd925), Yash Agarwal
**Course:** Advanced Deep Learning Systems 2024/25, Imperial College London  
**Framework:** MASE (DeepWok/mase) — PyTorch FX-based ML compiler  
**Repo:** https://github.com/aahaidar01/mase (branch: `feature/fused-rmsnorm-residual`)

### What This Does

Fuses two operations that run back-to-back in every transformer decoder layer into a single GPU kernel:

```
BEFORE (2 kernel launches, 1 redundant HBM round-trip):
    residual = residual + attn_output          ← writes x_a to HBM
    hidden   = RMSNorm(residual, weight)       ← reads x_a back from HBM

AFTER (1 kernel launch, x_a stays in registers):
    hidden, residual = FusedAddRMSNorm(residual, attn_output, weight, eps)
```

For Llama-2-7B (32 layers), this eliminates 32 redundant memory round-trips per forward pass.

### How It Fits in the Pipeline

```
ML Model (Llama / BERT / Mistral)
    ↓
Quantise Transform Pass (Part 3)
    ↓
FlexAttention Fusion (Part 1)
    ↓
Fused RMSNorm + Residual (Part 2)  ← this work
    ↓
LatencyRunner + Eval (Part 3)
    ↓
Optuna Search → Pareto Frontier
```

---

## 2. Development Timeline

### Phase 1: Standalone Kernel (Colab, T4 GPU)

**Step 1: Initial kernel implementation**
- Wrote `triton_fused_add_rmsnorm.py` with forward Triton kernel
- Each program instance processes one row (one token position) of (B×T, D) tensor
- BLOCK_SIZE = next_power_of_2(D), one tile per row, max 65536
- Three casting modes: llama (rstd in FP32), gemma (all FP32, offset 1.0), none

**Step 2: Backward kernel**
- Implemented `_fused_add_rmsnorm_bwd_kernel` computing dL/dX_residual, dL/dX_hidden, dL/dWeight
- Two-stage weight gradient reduction: per-row partials in Triton, summed across rows in PyTorch
- This avoids atomic operations in the kernel (follows Liger-Kernel pattern)

**Step 3: Python wrappers**
- `FusedAddRMSNorm` — torch.autograd.Function connecting kernels to PyTorch autograd
- `FusedAddRMSNormModule` — torch.nn.Module with learnable weight, drop-in replacement

**Step 4: Test harness**
- 144 forward tests: 8 shapes × 3 dtypes × 3 casting modes × 2 offsets
- 36 backward tests: gradient agreement with PyTorch reference
- nn.Module wrapper test
- Latency and memory benchmarks

**Step 5: BF16 numerical stability bug**
- **Problem:** 8/144 forward tests failed — all bf16 + casting_mode="none"
- **Root cause:** Sum-of-squares reduction done entirely in bf16. BF16 has ~3 decimal digits of precision; accumulating across 256–8192 elements caused significant drift
- **Fix:** Cast to FP32 before any reduction, even in "none" mode. Applied to forward kernel, backward kernel, and PyTorch reference
- **Result:** 144/144 forward + 36/36 backward all passing
- **Validation:** Matches Liger-Kernel's approach — they also accumulate in FP32 regardless

**Step 6: Initial benchmarks (T4)**

| Config | Unfused (µs) | Fused (µs) | Speedup |
|--------|-------------|-----------|---------|
| Batch inference, Llama-7B (bf16) | 740.7 | 150.7 | 4.92× |
| Long seq, Llama-7B (bf16) | 1235.1 | 289.0 | 4.27× |
| Peak memory (4,512,4096) bf16 | 80 MB | 32 MB | 60% reduction |

### Phase 2: Standalone Debug Scripts (Colab, L4 GPU)

Created two completely independent scripts for side-by-side comparison:

- `unfused_pytorch_baseline.py` — pure PyTorch, no Triton dependency
- `fused_triton_kernel.py` — self-contained Triton implementation

Both use identical seed (42), configs, and output format. Verified outputs match across all 6 configurations on L4 GPU. Confirmed speedups of 2.7× on batch workloads (L4 has higher bandwidth than T4, so smaller speedup margin but same behaviour).

### Phase 3: Kernel v2 + Expanded Tests (HPC, L40S GPU)

**Step 7: Kernel improvements**
- Updated warp-count heuristic to match Liger-Kernel/Unsloth thresholds (4/8/16/32 warps based on BLOCK_SIZE)
- Backward kernel refactored: llama/none modes now compute m = dNormed × w_eff in input dtype first, then upcast for reduction (more precise)
- Replaced CastingMode enum with simpler dict lookup

**Step 8: Expanded test suite**
- Test count increased from 144+36 to 180 total configurations
- Added 3-way comparison against Liger-Kernel (not just PyTorch baseline)
- All 180/180 tests passing on L40S

**Step 9: L40S benchmarks**

Kernel latency (3-way, CUDA events, BF16):

| Config | Unfused (µs) | Ours (µs) | Liger (µs) | vs Unfused | vs Liger |
|--------|-------------|-----------|-----------|-----------|---------|
| 4×512×4096 | 160 | 67 | 88 | **2.39×** | 1.31× |
| 8×512×4096 | 549 | 184 | 261 | **2.98×** | **1.42×** |
| 16×256×4096 | 543 | 183 | 257 | **2.97×** | **1.40×** |
| 8×256×8192 | 535 | 189 | 261 | **2.83×** | **1.39×** |

Peak memory (forward only, BF16):

| Config | Unfused | Fused | Reduction |
|--------|---------|-------|-----------|
| 4×512×4096 | 80.0 MB | 32.0 MB | 60.0% |
| 8×128×4096 | 40.0 MB | 16.0 MB | 60.0% |
| 2×128×8192 | 20.0 MB | 8.0 MB | 60.0% |

Forward+backward average memory reduction: 25.8%

### Phase 4: MASE Integration

**Step 10: Graph-level FX pass**
- Created `fused_rmsnorm_transform.py` — walks FX graph, pattern-matches add→RMSNorm nodes, replaces with FusedAddRMSNormModule
- Custom FX tracer (`_RMSNormLeafTracer`) keeps RMSNorm as leaf nodes instead of inlining
- Works on manually constructed models (1-block: PASS, 2-block: PASS, all casting modes: PASS)
- **Does NOT work on HuggingFace models** — FX tracer fails due to dynamic control flow (Proxy item assignment). This is a known limitation, not a bug

**Step 11: Module-level pass (primary integration)**
- Created `rmsnorm_residual_fusion.py` — walks model.named_modules(), patches decoder layer forward methods via types.MethodType
- No FX tracing required — works on all HuggingFace architectures
- Same monkey-patching approach used by Liger-Kernel in production
- Recognises: LlamaDecoderLayer, MistralDecoderLayer, GemmaDecoderLayer, Qwen2DecoderLayer, InternLMDecoderLayer

**Step 12: MASE registration**
- Registered both passes in MASE's pass system by modifying 4 `__init__.py` files:
  - `passes/graph/__init__.py` — added fused_rmsnorm_transform_pass to TRANSFORM_PASSES and PASSES dict
  - `passes/graph/transforms/__init__.py` — added import from fused_rmsnorm
  - `passes/module/__init__.py` — added fused_rmsnorm_residual_transform_pass
  - `passes/module/transforms/__init__.py` — added import from fused_ops

### Phase 5: End-to-End Validation (Inference)

**Step 13: Llama-2-7B standalone (HPC, L40S)**

| Metric | Value | Verdict |
|--------|-------|---------|
| Layers fused | 32/32 | ✅ |
| Max abs error | 1.05e-01 | — |
| Relative error (max/range) | 0.30% | ✅ (<1%) |
| Cosine similarity | 1.000000 | ✅ (>0.999) |
| Top-5 token match | Exact | ✅ |

Latency (CUDA events, 50 iterations):

| Config | Unfused | Fused | Speedup |
|--------|---------|-------|---------|
| B=1, seq=64 | 24.08 ms | 23.42 ms | 1.028× |
| B=1, seq=256 | 31.33 ms | 30.51 ms | 1.027× |
| B=4, seq=64 | 30.82 ms | 29.89 ms | 1.031× |

**Step 14: Module-level pass test (HPC, L40S)**

Run by Dorian on dd925 HPC account:

| Test | Result |
|------|--------|
| Tiny Llama (2 layers, random weights) | 2 fused, err=3.58e-07, top-5 match ✅ |
| Casting modes (llama/gemma/none) | All PASS, errors < 2.1e-07 ✅ |
| Llama-2-7B (32 layers) | 32 fused, rel_err=0.16%, cos_sim=1.000, top-5 match ✅ |
| Model latency | 24.08 → 23.36 ms (1.031×) ✅ |

**MODULE PASS: ALL TESTS PASSED**

### Phase 6: Training Validation (HPC, L40S)

**Step 15: Training loop tests**

Verified the fused kernel works correctly during training, not just inference:

| Test | What it proves | Result |
|------|---------------|--------|
| **Gradient flow** | All 21 parameters receive non-zero gradients, including all 4 RMSNorm weights | ✅ PASS |
| **Weight updates** | Optimizer.step() modifies all 12 weights including fused RMSNorm weights | ✅ PASS |
| **Overfitting** | Fused model memorises a single batch (loss 4.60 → 0.001, 100% reduction in 50 steps) — optimisation landscape is not broken | ✅ PASS |
| **Fused vs unfused training** | Two identical models (same init, same data) trained for 20 steps — loss trajectories match with max difference 9.54e-07. Final losses identical (0.3801) | ✅ PASS |

Key result from fused vs unfused comparison:
```
Step   Unfused Loss     Fused Loss       Diff
0      6.946591         6.946591         0.00e+00
1      6.097987         6.097987         0.00e+00
4      4.376801         4.376801         0.00e+00
19     0.380118         0.380118         5.96e-08
```

The fused kernel is numerically transparent during training — it produces the same optimisation trajectory as unfused PyTorch.

**TRAINING TESTS: ALL PASSED**

### Phase 7: GitHub Push + Integration

**Step 16: Pushed to GitHub**
- Branch: `feature/fused-rmsnorm-residual` on `github.com/aahaidar01/mase`
- 5 new files, 4 modified `__init__.py` files
- All committed and pushed

**Step 17: L40S plots generated**
- Regenerated all benchmark plots using L40S data (not T4/L4)
- Log-log scaling plot cropped to ≥4M elements to show relevant operating range
- Three plots: latency+speedup (BF16), log-log scaling, all-config speedup (BF16+FP16+FP32)

---

## 3. File Inventory

### In MASE repo (`feature/fused-rmsnorm-residual` branch)

| File | Purpose |
|------|---------|
| `src/chop/passes/graph/transforms/fused_rmsnorm/__init__.py` | Package exports (graph pass) |
| `src/chop/passes/graph/transforms/fused_rmsnorm/triton_fused_add_rmsnorm.py` | Triton fwd/bwd kernels, autograd Function, nn.Module |
| `src/chop/passes/graph/transforms/fused_rmsnorm/fused_rmsnorm_transform.py` | Graph-level FX transform pass |
| `src/chop/passes/module/transforms/fused_ops/__init__.py` | Package exports (module pass) |
| `src/chop/passes/module/transforms/fused_ops/rmsnorm_residual_fusion.py` | Module-level transform pass (primary) |
| `src/chop/passes/graph/__init__.py` | Modified — registered graph pass |
| `src/chop/passes/graph/transforms/__init__.py` | Modified — added fused_rmsnorm import |
| `src/chop/passes/module/__init__.py` | Modified — registered module pass |
| `src/chop/passes/module/transforms/__init__.py` | Modified — added fused_ops import |

### HPC standalone files (`hpc_run/`)

| File | Purpose |
|------|---------|
| `triton_fused_add_rmsnorm.py` | Core kernel (standalone copy) |
| `fused_rmsnorm_transform.py` | Graph pass (standalone copy) |
| `test_fused_add_rmsnorm.py` | 180 kernel correctness tests |
| `test_module_pass.py` | Module-level pass test (tiny + casting + 7B) |
| `test_mase_transform.py` | Graph-level pass test |
| `test_7b_model.py` | Llama-2-7B end-to-end standalone |
| `test_training.py` | Training validation (gradient flow, weight updates, overfitting, fused vs unfused) |
| `benchmark_3way.py` | 3-way latency comparison |
| `benchmark_memory.py` | Peak memory benchmark |
| `run_all.py` | Orchestrator for all tests |
| `run_tests.pbs` | PBS: kernel correctness |
| `run_benchmarks_only.pbs` | PBS: all benchmarks |
| `run_mase_transform.pbs` | PBS: graph-level pass |
| `run_module_pass.pbs` | PBS: module-level pass |
| `run_7b.pbs` | PBS: Llama-2-7B standalone |
| `run_training.pbs` | PBS: training validation |

### HPC outputs (`hpc_outputs/`)

| File | Contents |
|------|----------|
| `fused_rmsnorm_tests.out` | 180/180 kernel tests + 3-way benchmark PASS |
| `fused_rmsnorm_bench1.out` | Full latency + memory + 3-way benchmarks (L40S) |
| `fused_rmsnorm_mase.out` | Graph-level pass tests (manual PASS, HF SKIPPED) |
| `fused_rmsnorm_7b.out` | Llama-2-7B end-to-end PASS (L40S) |
| `fused_rmsnorm_module.out` | Module pass ALL TESTS PASSED (L40S) |
| `fused_rmsnorm_training.out` | Training validation ALL PASSED (L40S) |

---

## 4. Technical Design

### RMSNorm Formula

```
residual_out  = X_residual + X_hidden
rms           = sqrt( (1/D) * sum(residual_out²) + eps )
normed_out    = (residual_out / rms) * (weight + offset)
```

Where offset = 0.0 for Llama/Mistral, 1.0 for Gemma.

### RMSNorm Backward

```
Let r = residual, rstd = 1/RMS(r), w = weight + offset

dL/dr = rstd * (dNormed * w) - (rstd³ / D) * sum(dNormed * w * r) * r
dL/dWeight = sum_over_rows(dNormed * r * rstd)

Since residual = X_res + X_hid:
    dL/dX_res = dL/dr + dL/d(residual_out)
    dL/dX_hid = dL/dr + dL/d(residual_out)
```

### Casting Modes

| Mode | Behaviour | Used by |
|------|-----------|---------|
| `"llama"` | Only rstd computed in FP32; normalisation in input dtype | Llama 2/3, Mistral |
| `"gemma"` | Everything cast to FP32; weight offset = 1.0 | Gemma, Gemma 2 |
| `"none"` | FP32 for reductions only; rest in input dtype | General |

### Kernel Design Choices

- **One row per program instance** — each Triton program handles one token position
- **Single BLOCK_SIZE tile** — next_power_of_2(hidden_dim), max 65536
- **FP32 accumulation for all reductions** — even in "none" mode (learned from BF16 bug)
- **Two-stage weight gradient reduction** — per-row partials in Triton, summed in PyTorch
- **num_warps heuristic** — 4 (<2048), 8 (2048–8191), 16 (8192–32767), 32 (≥32768)

### Why Two MASE Passes

| Pass | Approach | Works on HF models? | How it works |
|------|----------|---------------------|-------------|
| Graph-level | FX pattern matching | No (tracer breaks) | Traces model to FX graph, finds add→RMSNorm node pairs, rewrites graph |
| Module-level | Monkey-patching | **Yes** | Walks named_modules(), patches forward() of decoder layers |

The graph-level pass is theoretically cleaner but HuggingFace models use dynamic control flow (`Proxy` object does not support item assignment) that prevents FX tracing. The module-level pass sidesteps this entirely — same approach Liger-Kernel uses in production.

---

## 5. Integration with Parts 1 & 3

### Branch

```
Repo:   https://github.com/aahaidar01/mase
Branch: feature/fused-rmsnorm-residual
```

### For teammates

```bash
git fetch origin
git checkout feature/fused-rmsnorm-residual
# or merge into your branch:
git merge origin/feature/fused-rmsnorm-residual
```

### How Part 3 calls Part 2

```python
from chop.passes.module.transforms.fused_ops import fused_rmsnorm_residual_transform_pass

model, info = fused_rmsnorm_residual_transform_pass(model, {
    "casting_mode": "llama",
})
# info["num_fused"] = 32 for Llama-2-7B
# info["fused_layers"] = ["model.layers.0", "model.layers.1", ...]
```

### In the Optuna search config

```python
fusion_strategy = trial.suggest_categorical(
    "fusion_strategy", ["none", "flex_attention", "fused_rmsnorm", "both"]
)

if fusion_strategy in ("fused_rmsnorm", "both"):
    model, _ = fused_rmsnorm_residual_transform_pass(model, {"casting_mode": "llama"})
```

### Pass return convention

Both passes return `(model, dict)` following MASE convention:
- Module pass: `return network, {"num_fused": N, "fused_layers": [...]}`
- Graph pass: `return graph, {"num_fused": N}`

---

## 6. HPC Environment

| Detail | Value |
|--------|-------|
| Login | `dd925@login.cx3.hpc.imperial.ac.uk` |
| Working dir | `/rds/general/user/dd925/home/ADLS_Project/hpc_run` |
| Conda | `~/miniforge3`, env name `triton_env` |
| Python | 3.11 |
| PyTorch | 2.7.1+cu118 |
| Triton | 3.3.1 |
| Transformers | 5.3.0 |
| GPU | NVIDIA L40S (46 GB) |
| HF token | Configured via huggingface_hub login |

### To reconnect and run jobs

```bash
ssh dd925@login.cx3.hpc.imperial.ac.uk
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate triton_env
cd ~/ADLS_Project/hpc_run
qsub run_training.pbs          # training validation
qsub run_module_pass.pbs       # module pass (inference)
qsub run_tests.pbs             # 180 kernel tests
qsub run_benchmarks_only.pbs   # latency + memory benchmarks
qstat -u dd925                 # check status
```

---

## 7. Complete Results Summary

### Kernel Correctness
- **180/180 tests PASSED** on L40S
- 3 dtypes (BF16, FP16, FP32) × 3 casting modes × 20 shapes
- Forward + backward + nn.Module wrapper all verified

### Kernel Latency (L40S, BF16, CUDA events)
- **Up to 2.98× faster** than unfused PyTorch
- **Up to 1.42× faster** than Liger-Kernel
- Crossover at ~2048 rows; below this, launch overhead dominates

### Peak Memory
- **60% reduction** per fusion site (forward only)
- **25.8% average** during forward+backward
- Saving = exactly one (B,T,D) tensor eliminated per site

### Llama-2-7B End-to-End Inference (L40S)
- 32/32 layers fused automatically
- Relative error: 0.30% (well within 1% threshold)
- Cosine similarity: 1.000000
- Top-5 predictions: exact match
- Model-level speedup: 1.03× (modest — add+RMSNorm is <5% of forward time)
- Principal benefit: memory pressure reduction enabling larger batch sizes

### Training Validation (L40S)
- All parameters receive gradients including fused RMSNorm weights ✅
- Optimizer updates all weights correctly ✅
- Fused model overfits single batch: loss 4.60 → 0.001 (100% reduction) ✅
- Fused vs unfused training trajectories match within 9.54e-07 over 20 steps ✅
- **Fused kernel is numerically transparent during training**

### MASE Integration
- Graph pass: works on manual models (PASS), skips HF models (expected)
- Module pass: **ALL TESTS PASSED** on L40S including Llama-2-7B
- Both passes registered in MASE's pass system
- Pushed to GitHub on `feature/fused-rmsnorm-residual`

---

## 8. Limitations and Future Work

| Limitation | Detail |
|-----------|--------|
| Only 1 of 2 sites fused | Post-attention only; post-MLP requires cross-layer coordination |
| Small-input overhead | Below ~2048 rows, fused kernel is ~3% slower due to launch cost |
| FP32 no benefit | PyTorch native kernels already saturate bandwidth at 4 bytes/element |
| FX tracer incompatibility | HuggingFace models need module-level pass (graph pass skips them) |
| Mistral-7B untested | Failed due to transformers version incompatibility |

### Potential extensions
- Fuse the post-MLP site (second residual add + next layer's input_layernorm)
- Triton autotune across block sizes and warp counts for different GPU architectures
- Extend to LayerNorm for BERT-family models
- Add the fused kernel as a configurable option in MASE's CLI

---

## 9. Reference Implementations

| Reference | How it was used |
|-----------|----------------|
| **Liger-Kernel** (linkedin/Liger-Kernel) | Primary reference for RMSNorm kernel design, casting modes, backward pass derivation, two-stage weight gradient pattern |
| **Unsloth** (unslothai/unsloth) | Secondary reference for Triton RMSNorm patterns and warp-count heuristics |
| **MASE** (DeepWok/mase) | Framework conventions for transform passes, MaseGraph, pass registration |
