# Lab Results Comparison: My Results vs Friends' Analysis

## Lab 1: Model Compression

### Task 1: Quantization (PTQ vs QAT)

**My Results:**

| Bit Width | PTQ | QAT | Gap |
|-----------|-----|-----|-----|
| 4 | 0.5000 | 0.5000 | 0.0000 |
| 6 | 0.6894 | 0.8233 | +0.1340 |
| 8 | 0.7332 | 0.8328 | +0.0995 |
| 12 | 0.8088 | 0.8347 | +0.0259 |
| 16 | 0.8137 | 0.8350 | +0.0214 |
| 24 | 0.8141 | 0.8346 | +0.0205 |
| 32 | 0.8139 | 0.8346 | +0.0207 |

- FP32 baseline: 0.8140

**Comparison: STRONG AGREEMENT**

Similarities:
- Both observe QAT consistently outperforms PTQ across all bit widths.
- Both show the gap is largest at low bit widths: my 6-bit gap is +13.4%, aligning with friends' observation that "PTQ simply rounds existing high-precision weights, while QAT retrains the model to actively adapt to the significant, non-linear noise introduced by aggressive quantization."
- Both see 4-bit as a total failure point (0.50 = chance level), where quantization is too aggressive for either method.
- Both confirm diminishing returns at higher bit widths -- my PTQ plateaus at ~0.814 from 16 bits onwards, matching the FP32 baseline.

Differences:
- Friends explored integer vs fractional width separately (finding "not much benefit beyond 5 integer bits with 4 fractional bits"). I now also ran an imbalanced precision sweep across {8, 12, 16}-bit total widths with varying frac_width, confirming their finding: with only 2 fractional bits, accuracy collapses to 0.50 regardless of total width. PTQ peaks at balanced allocations (frac ~ total/2), while QAT is robust once frac >= 4 bits.
- My QAT results (~0.835 at 8+ bits) actually **exceed the FP32 baseline** (0.814), suggesting QAT acts as a regulariser. Friends' writeup doesn't explicitly highlight this QAT-as-regularisation effect.
- At 6-bit, the PTQ-QAT gap (+13.4%) is extremely large, strongly supporting the theoretical explanation about quantization noise.
- My imbalanced sweep shows diminishing returns from extra integer bits: at frac=4, QAT accuracy is ~0.833 whether total width is 8, 12, or 16. The bottleneck is fractional precision, not dynamic range.

---

### Task 2: Pruning (Random vs L1-Norm)

**My Results (expanded 9-level sweep):**

| Sparsity | Random | L1-Norm | Gap |
|----------|--------|---------|-----|
| 10% | 0.8200 | 0.8409 | +0.0209 |
| 20% | 0.7961 | 0.8366 | +0.0404 |
| 30% | 0.7626 | 0.8322 | +0.0696 |
| 40% | 0.6027 | 0.8263 | +0.2236 |
| 50% | 0.5149 | 0.8151 | +0.3001 |
| 60% | 0.5052 | 0.8023 | +0.2972 |
| 70% | 0.4990 | 0.7535 | +0.2546 |
| 80% | 0.4988 | 0.6073 | +0.1085 |
| 90% | 0.5049 | 0.5363 | +0.0314 |

- Baseline (pre-pruning): 0.83504

**Comparison: STRONG AGREEMENT**

Similarities:
- Both confirm L1-norm consistently outperforms random pruning at every sparsity level.
- Both show random pruning collapses to chance (~0.50) at higher sparsity -- my random at 50% (0.515) and 70% (0.499) match their description of "random pruning essentially collapses."
- L1-norm degrades gracefully in both cases -- mine goes from 0.841 -> 0.832 -> 0.815 -> 0.754 -> 0.607.
- Both now cover the full 10%-90% range, allowing identification of the same regimes: safe (10-30%), transition (40-60%), and collapse (70-90%) for random; safe (10-60%), transition (70-80%), and collapse (90%) for L1-norm.

Differences:
- L1-norm at 10% sparsity **exceeds the baseline** (0.841 > 0.835), a regularisation effect not highlighted in friends' writeup. Removing the smallest weights acts as beneficial noise reduction.
- Random pruning cliff occurs sharply between 30% (0.763) and 40% (0.603) in my results. Friends describe 40-50% as the "sharper accuracy drop" zone -- this aligns well, with my cliff perhaps slightly earlier due to BERT-tiny's small parameter count.
- My baseline of 0.835 (from the QAT model) vs their FP32 baseline may differ, since they built on "the best trained, quantized model from the previous tutorial."

---

## Lab 2: Neural Architecture Search

### Task 1: NAS Sampler Comparison (100 trials)

**My Results:**

| Sampler | Best Accuracy | Trial Found Best |
|---------|---------------|-----------------|
| Grid | 0.87232 | Trial 27 |
| Random | 0.87232 | Trial 16 |
| TPE | 0.87232 | Trial 2 |

**Comparison: PARTIAL AGREEMENT**

Similarities:
- TPE finds the optimal configuration fastest -- my TPE found 0.87232 on trial 2, while Grid needed 27 trials and Random needed 16. This strongly confirms their observation that "TPE rapidly concentrates evaluations in high-performing regions."
- The same best accuracy ceiling of 0.87232 appears across all samplers, confirming the search space has a clear global optimum.

Differences:
- Friends focused on Grid vs TPE comparison, while I also included Random sampler.
- **Key difference:** My Grid sampler achieves the same best accuracy as TPE (0.87232), while friends describe Grid as having "high variance and poor sample efficiency." In my results, with 100 trials Grid finds the best, just slower. The search space may be small enough that even an exhaustive approach eventually finds it.
- Friends' writeup implies TPE achieves a *higher* final accuracy, but in my data all three samplers converge to the identical 0.87232. This is likely because the discrete search space (num_layers, num_heads, hidden_size, intermediate_size) has a finite number of configurations, so 100 trials is enough for all samplers to hit the best.

---

### Task 2: Compression-Aware NAS (10 trials)

**My Results:**

| Condition | Best Accuracy | Variance |
|-----------|--------------|----------|
| Baseline | 0.86624 | Low |
| Compress-Only | 0.86188 | Very High (0.597-0.862) |
| Compress+Finetune | **0.87644** | Very Low (0.870-0.876) |

**Comparison: STRONG AGREEMENT**

Similarities:
- Both confirm Compress+Finetune produces the best results: my 0.87644 > Baseline 0.86624.
- Both show Compress-Only has very high variance -- my results range from 0.597 to 0.862.
- Both agree that post-compression training recovers and sometimes exceeds the baseline, supporting the "improved generalisation" explanation.

Differences:
- I ran 10 trials per condition vs their unspecified count.
- My Compress+Finetune has remarkably low variance (0.870-0.876), showing the approach is robust. Friends don't quantify this stability.
- My Compress-Only best (0.86188) is close to baseline (0.86624), but the median is much lower due to extreme failures like 0.597 -- more extreme variance than friends describe.

---

## Lab 3: Mixed Precision Search

### Task 1: Per-Layer Mixed Precision Search (20 trials)

#### Bug Discovery and Fix

The original code had a critical bug in `get_integer_config()` that caused 70% of trials to crash. The `suggest_categorical` for `frac_width` used **dynamic choice sets** that depended on the sampled `width`:

```python
# BUGGY: choices change between trials, violating Optuna's constraint
valid_data_frac = [f for f in FRAC_WIDTH_CHOICES if f < data_in_width]
```

For example, if Trial 0 registered `query_data_in_frac_width` with choices `[2, 4]` (because `data_in_width=8`), and Trial 4 later sampled `data_in_width=16` which needs choices `[2, 4, 8]`, Optuna raises a `ValueError` because categorical parameters must have consistent distributions across trials. The `except Exception` block silently caught this and returned 0.0.

**Fix applied:** Use fixed choice sets and clamp invalid combos afterward:

```python
# FIXED: static choices, Optuna-safe
data_in_frac_width = trial.suggest_categorical(
    f"{layer_name}_data_in_frac_width", FRAC_WIDTH_CHOICES
)
if data_in_frac_width >= data_in_width:
    data_in_frac_width = data_in_width // 2
```

#### Results Comparison: Old (Buggy) vs New (Fixed)

| Metric | Old (buggy) | New (fixed) |
|--------|-------------|-------------|
| Crashed trials (0.0) | 14 (70%) | 0 (0%) |
| Chance trials (~0.5) | 0 | 1 (5%) |
| Valid trials | 6 (30%) | 19 (95%) |
| Best accuracy | 0.85816 | **0.85980** |
| Best found at trial | 12 | 6 |
| Mean accuracy (valid) | 0.85488 | 0.85459 |

The single remaining chance-level trial (Trial 2, 0.5) is a genuinely degenerate quantization config, not an Optuna crash.

#### Best Configuration (Trial 6, accuracy 0.85980)

| Layer | Quantization Type |
|-------|-------------------|
| layer.0.attn.self.query | Linear (FP32) |
| layer.0.attn.self.key | Linear (FP32) |
| layer.0.attn.self.value | Linear (FP32) |
| layer.0.attn.output.dense | LinearInteger |
| layer.0.intermediate.dense | Linear (FP32) |
| layer.0.output.dense | LinearMinifloatIEEE |
| layer.1.attn.self.query | LinearInteger |
| layer.1.attn.self.key | LinearMinifloatIEEE |
| layer.1.attn.self.value | Linear (FP32) |
| layer.1.attn.output.dense | LinearMinifloatIEEE |
| layer.1.intermediate.dense | LinearInteger |
| layer.1.output.dense | LinearMinifloatIEEE |
| bert.pooler.dense | Linear (FP32) |
| classifier | LinearMinifloatIEEE |

The best config keeps 6 out of 14 layers at full precision, notably all of layer 0's attention self-projections (Q, K, V). This suggests the first encoder layer is more sensitive to quantization than the second.

**Comparison with Friends: AGREEMENT (after fix)**

Similarities:
- Both observe mixed precision achieves slightly lower accuracy than uniform quantization from Lab 2. My best 0.860 vs Lab 2's 0.872 confirms their finding.
- Both attribute this to "the optimal hyperparameter search space being more complicated."
- After the fix, my failure rate (0% crashes, 5% degenerate) now matches what friends likely saw.

Differences:
- My old 70% failure rate was entirely due to the Optuna bug, not an inherent property of mixed precision search. Friends likely had correct code and never saw this.
- With the fix, best accuracy improved from 0.858 to 0.860 -- the bug was limiting search effectiveness by wasting 70% of trials on crashes, giving TPE fewer data points to learn from.
- The gap from uniform quantization (0.860 vs 0.872 = 1.4%) is a genuine finding about search space complexity.

---

### Task 2: Mixed Precision Sampler Comparison (100 trials per sampler)

**My Results:**

| Sampler | Best Accuracy | Trials at ~0.5 |
|---------|---------------|----------------|
| Random | 0.86036 | ~8 |
| TPE | 0.86052 | ~4 |

**Comparison: PARTIAL AGREEMENT**

Similarities:
- Both samplers achieve similar best accuracies, consistent with friends' experiments.
- The general finding that mixed precision with per-layer types is achievable is shared.

Differences:
- My Random (0.86036) vs TPE (0.86052) gap is only 0.00016 -- essentially identical even after 100 trials each. This is much closer than friends' results suggest.
- Both samplers have trials that collapse to 0.5 (degenerate configs). TPE has fewer (4 vs 8), suggesting it learns to avoid bad configurations.
- Friends' analysis of uniform per-layer precision types (Integer, Log, Minifloat, Binary etc.) is a separate experiment I didn't run -- they found Binary collapses completely while Integer/Log/BlockFP preserve accuracy.

---

## Overall Conclusions

### Key Agreements

1. **QAT > PTQ** at all bit widths, with the gap widening at lower precision -- universal finding.
2. **L1-norm > Random pruning** across all sparsities -- both confirm structured removal outperforms random removal.
3. **TPE is the most sample-efficient sampler** -- finds optima faster even if all samplers eventually converge given enough trials.
4. **Compression-aware NAS with finetuning beats all other approaches** (0.876) -- the most actionable finding across all labs.
5. **Mixed precision search is harder than uniform quantization** -- larger search space with degenerate configurations, and 1-2% lower accuracy.

### Key Differences Worth Discussing

1. **NAS samplers converge to identical best accuracy** (0.87232) at 100 trials in my results, while friends imply TPE reaches a higher ceiling. This likely reflects the discrete, bounded search space -- with enough trials, even Grid/Random find the optimum.
2. **Lab 3 Task 1 had a code bug** (dynamic `suggest_categorical` choices) causing 70% crash rate. After fixing, results align with friends' analysis. This is a known Optuna pitfall worth noting.
3. **Pruning shows earlier random collapse** in my results (cliff at 40% sparsity, chance by 50%), possibly due to the smaller BERT-tiny model having less redundancy. The expanded 9-level sweep now matches friends' granularity and confirms the same regime structure.
4. **QAT exceeds FP32 baseline** in my results (~0.835 vs 0.814) -- a regularisation effect worth highlighting if friends didn't observe it.
5. **L1-norm pruning at 10% exceeds baseline** (0.841 > 0.835) -- another regularisation effect, where removing near-zero weights reduces noise. This complements the QAT regularisation finding.
6. **Imbalanced precision sweep** confirms friends' finding that fractional bits matter more than integer bits: with frac=2, accuracy collapses regardless of total width.

### Accuracy Hierarchy Across All Labs

| Approach | Best Accuracy | Lab |
|----------|--------------|-----|
| Compress+Finetune NAS | **0.87644** | Lab 2 Task 2 |
| Architecture NAS (any sampler) | 0.87232 | Lab 2 Task 1 |
| Baseline NAS (no compression) | 0.86624 | Lab 2 Task 2 |
| Mixed Precision Sampler Search | 0.86052 | Lab 3 Task 2 |
| Mixed Precision TPE Search | 0.85980 | Lab 3 Task 1 |
| QAT (16-bit) | 0.83504 | Lab 1 Task 1 |
| FP32 Baseline (no optimisation) | 0.81396 | Lab 1 Task 1 |
| L1-Norm Pruning (10% sparse) | 0.84088 | Lab 1 Task 2 |

The clear winner is compression-aware NAS with post-compression finetuning, which combines the benefits of architecture search with the regularisation effect of quantization.
