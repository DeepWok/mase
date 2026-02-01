# Lab 2 – Tutorial 5: Neural Architecture Search with Optuna and Compression-Aware Optimisation

## Objective
Investigate how different **search strategies** explore the model design space for BERT-based architectures, and evaluate how **compression-aware search** affects final model quality. The goal is to understand:

1. How **GridSampler** and **TPESampler** compare to random search in NAS.
2. Whether architectures found via NAS remain optimal after **quantisation and pruning**.
3. Whether incorporating compression *during* search yields better post-compression models.

---

## Baseline NAS Setup (Task 1)

### Search Space
The NAS search space included:
- **Model-level hyperparameters**
  - `num_layers`
  - `num_heads`
  - `hidden_size`
  - `intermediate_size`
- **Structural choices**
  - Optional replacement of square `Linear` layers with `Identity`

Each trial:
1. Constructed a BERT-like model from a sampled configuration.
2. Trained for **1 epoch** on IMDb.
3. Evaluated classification accuracy.

This limited training budget is intentional and standard in NAS, allowing many trials to be compared cheaply.

---

## Samplers Compared

### GridSampler
- Exhaustively enumerates a **fixed Cartesian product** of hyperparameters.
- Requires all parameters to be known *a priori*.
- Cannot support dynamic, layer-specific parameters.

To enable GridSampler, the search was restricted to **model-level hyperparameters only**.

### TPESampler (Tree-structured Parzen Estimator)
- A **Bayesian optimisation** method.
- Builds probabilistic models of *good* vs *bad* regions of the search space.
- Supports **conditional and dynamic parameters**, making it suitable for architecture-level NAS.

---

## Task 1 Results: Sampler Comparison

A plot was generated with:
- **x-axis**: number of trials
- **y-axis**: best accuracy achieved up to that trial

### Observations
- **GridSampler** improved steadily but slowly.
- **TPESampler** achieved higher accuracy in fewer trials.
- TPESampler showed better *sample efficiency*, reaching strong configurations earlier.

### Interpretation
Grid search treats all configurations equally, while TPE actively exploits promising regions of the space. This confirms that:

> **Bayesian optimisation is more effective than exhaustive or random search in high-dimensional NAS problems.**

TPESampler was therefore selected for subsequent compression-aware experiments.

---

## Compression-Aware NAS (Task 2)

### Motivation
In earlier tutorials, NAS and compression were treated as **separate steps**:
1. Search for the best architecture.
2. Compress it afterward.

However, different architectures exhibit **different sensitivities to quantisation and pruning**. A model that performs best *before* compression may not be optimal *after* compression.

### Compression Pipeline
Each trial applied:
- **Quantisation**:
  - Integer quantisation
  - 8-bit data, weights, and bias
- **Pruning**:
  - 50% sparsity
  - L1-norm based pruning
  - Local scope (per-layer)

---

## Compression-Aware Objective Variants

Three experiment variants were run:

1. **Baseline NAS (no compression)**  
   Standard TPESampler NAS results from Task 1.

2. **Compression-aware NAS (no post-training)**  
   - Train model
   - Apply quantisation and pruning
   - Evaluate immediately

3. **Compression-aware NAS (with post-training)**  
   - Train model
   - Apply quantisation and pruning
   - Fine-tune for an additional epoch
   - Evaluate

---

## Task 2 Results

A second plot was generated with:
- **x-axis**: number of trials
- **y-axis**: best accuracy achieved up to that trial
- **Three curves**:
  - Baseline NAS
  - Compression-aware (no post-training)
  - Compression-aware (with post-training)

### Observations
- Applying compression **without post-training** caused a noticeable drop in accuracy.
- Compression-aware search **with post-training** recovered much of the lost performance.
- In some cases, compression-aware models approached or matched baseline NAS accuracy.

### Interpretation
These results demonstrate that:

- Compression significantly alters model behaviour.
- Architectures must adapt to quantisation and pruning effects.
- **Post-compression training is critical** to regain performance.
- Compression-aware NAS produces models that are more robust to deployment constraints.

---

## Key Takeaways

- TPESampler outperforms GridSampler and random search for NAS due to its adaptive, probabilistic search strategy.
- Separating NAS and compression is suboptimal.
- Compression-aware search yields better final models than compressing after NAS.
- Fine-tuning after compression is essential, especially under aggressive quantisation and pruning.

---

## Final Insight

Neural architecture search should not be viewed in isolation. When deployment constraints such as quantisation and sparsity matter, they must be **co-optimised during search**, not applied as an afterthought.

This tutorial demonstrates how MASE enables such **end-to-end, system-aware optimisation workflows**.

## Why Compression-Aware NAS Matters

A key insight from this tutorial is that *model quality is not absolute* — it depends on the constraints under which the model will be deployed.

In Task 1, architectures were evaluated assuming full-precision, dense computation. However, in realistic deployment scenarios, models are often subject to:
- Quantised arithmetic
- Weight sparsity
- Limited numerical precision

Task 2 demonstrates that architectures which perform best *before* compression are not necessarily optimal *after* compression. Compression-aware NAS exposes each candidate architecture to quantisation and pruning during search, allowing Optuna to favour architectures that are intrinsically more robust to these transformations.

This explains why:
- Compression without post-training causes large accuracy drops
- Post-compression fine-tuning restores performance
- Compression-aware NAS can approach baseline accuracy despite deployment constraints

The takeaway is that **architecture search, numerical representation, and training dynamics must be co-optimised**, rather than treated as independent steps.

