# Tutorial 6 – Mixed-Precision Quantisation Search

## Objective

The objective of Tutorial 6 was to investigate **mixed-precision quantisation** as a *search problem*, rather than a fixed post-training decision. Instead of uniformly quantising all layers, we used **Optuna** to determine **which individual layers should remain in full precision and which should be quantised**, allowing the model to balance accuracy and efficiency.

This tutorial extends Tutorial 5 by:
- Fixing the architecture (BERT-tiny),
- Performing **layer-wise precision search**,
- Evaluating robustness to **subsequent pruning and post-compression training**.

---

## Methodology

### Mixed-Precision Search Setup

- **Base model**: BERT-tiny (sequence classification on IMDb)
- **Search variable**: For *each Linear layer*, choose between:
  - Full precision (`torch.nn.Linear`)
  - Integer quantised (`LinearInteger`, 8-bit)
- **Search method**: Optuna with **TPESampler**
- **Training budget per trial**: 1 epoch
- **Number of trials**: 12

Each trial:
1. Constructed a mixed-precision model according to Optuna’s suggestions.
2. Trained the model briefly.
3. Evaluated classification accuracy.
4. Returned accuracy as the optimisation objective.

---

## Mixed-Precision Search Results

### Best Search Performance

- **Best validation accuracy during search**: **0.8766**
- **Sampler**: TPESampler
- **Number of trials**: 12

The running-best curve shows that:
- Accuracy improved rapidly in early trials,
- Converged after ~7 trials,
- TPESampler efficiently identified high-performing precision patterns.

---

## Best Mixed-Precision Configuration

The best-performing model used a **non-uniform precision layout**:

### Observed Precision Pattern

- **Attention query projections** remained **full precision**
- **Key/value projections** were often quantised
- **Feed-forward layers** were largely quantised
- **Classifier head** was quantised

This indicates that:
- Some layers (e.g. attention queries) are **more sensitive to quantisation**
- Other layers tolerate aggressive precision reduction with minimal impact

This validates the motivation for **mixed-precision search** over uniform quantisation.

---

## Post-Search Evaluation (Before Compression)

When the best mixed-precision model was evaluated *without further compression*:

- **Evaluation accuracy**: **0.8246**

The drop from the search objective is expected due to:
- Short training budgets during NAS
- Evaluation on a held-out dataset

---

## Compression Experiments

After selecting the best mixed-precision model, we applied **Mase’s CompressionPipeline**:

### Compression Settings

- **Quantisation**: Integer (8-bit)
- **Pruning**:
  - Sparsity: **50%**
  - Method: L1-norm
  - Scope: Local (per-layer)

Two variants were evaluated:

---

### 1. Compression Without Post-Training

- **Accuracy**: **0.502**

This large drop demonstrates that:
- Compression significantly perturbs learned representations
- Quantisation + pruning introduces substantial error without adaptation

---

### 2. Compression With Post-Training Fine-Tuning

- **Post-training epochs**: 1
- **Accuracy**: **0.86296**

Post-training recovered **most of the lost accuracy**, approaching the pre-compression search performance.

---

## Key Observations

1. **Mixed-precision matters**  
   Different layers have different sensitivity to quantisation. Allowing per-layer decisions improves overall robustness.

2. **Compression alone is insufficient**  
   Applying quantisation and pruning without retraining leads to severe degradation.

3. **Post-compression training is critical**  
   Even a single fine-tuning epoch substantially restores performance.

4. **Search + compression must be co-designed**  
   Models optimised with awareness of deployment constraints outperform naïvely compressed models.

---

## Final Insight

Tutorial 6 demonstrates that **precision itself is a tunable architectural parameter**. By integrating mixed-precision decisions into the search loop, we can discover models that:

- Maintain high accuracy,
- Are significantly more efficient,
- Are robust to aggressive pruning and quantisation.

This tutorial completes the progression from:
- **Architecture search (Tutorial 5)**  
→ **Compression-aware optimisation**  
→ **Mixed-precision, deployment-aware model design**

and highlights the importance of **end-to-end system-aware optimisation** in modern ML pipelines.
