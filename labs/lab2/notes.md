# Lab 2 – Tutorial 5: Neural Architecture Search and Compression-Aware Optimisation

## Objective

The aim of this lab is to understand how different **search strategies** explore neural network architectures, and how **model compression** affects the final performance of a model.

In particular, this lab looks at:

1. How **GridSampler** and **TPESampler** behave when used for Neural Architecture Search (NAS)
2. Whether architectures found by NAS still perform well after **quantisation and pruning**
3. Whether applying compression **during** the search process leads to better final results

---

## Baseline NAS Setup (Task 1)

### Search Space

The architecture search was carried out over the following parameters:

- **Model-level parameters**
  - `num_layers`
  - `num_heads`
  - `hidden_size`
  - `intermediate_size`

- **Structural choices**
  - Some square `Linear` layers could be replaced with `Identity` layers

For each trial, the following steps were performed:

1. A BERT-like model was built using the sampled parameters
2. The model was trained for **1 epoch** on the IMDb dataset
3. Classification accuracy was evaluated

Training for only one epoch is intentional. In NAS, the goal is to quickly compare many architectures rather than fully training each model.

---

## Samplers Compared

### GridSampler

- Tries **all possible combinations** of parameters in a fixed grid
- Requires the full set of parameters to be known **before** the search starts
- Cannot support parameters that depend on the model structure, such as per-layer choices

Because of this, GridSampler was only used for **model-level parameters**.  
Layer-specific decisions were not included when using GridSampler.

---

### TPESampler (Tree-structured Parzen Estimator)

- Uses results from previous trials to guide future sampling
- Focuses more on parameter values that have performed well so far
- Supports **dynamic and conditional parameters**, such as layer-by-layer decisions

This makes TPESampler much more suitable for real NAS problems, where the model structure can change between trials.

---

## Task 1 Results: Sampler Comparison

A plot was generated with:

- **x-axis**: number of trials
- **y-axis**: best accuracy achieved up to that point

### Observations

- GridSampler showed slow but steady improvement
- TPESampler reached higher accuracy in fewer trials
- TPESampler found strong architectures earlier in the search

### Interpretation

GridSampler treats all configurations equally, while TPESampler learns which choices are more promising.  
This shows that:

> **TPESampler is more efficient than grid search or random search for large architecture spaces.**

For this reason, TPESampler was chosen for the next task.

---

## Compression-Aware NAS (Task 2)

### Motivation

In earlier tutorials, architecture search and compression were handled as two separate steps:

1. Find the best architecture
2. Apply compression afterward

However, different architectures respond differently to compression.  
A model that performs well before compression may lose a lot of accuracy after quantisation or pruning.

---

## Compression Pipeline

In Task 2, compression was included directly in the evaluation of each trial:

- **Quantisation**
  - Integer quantisation
  - 8-bit precision for inputs, weights, and bias

- **Pruning**
  - 50% sparsity
  - L1-norm based pruning
  - Applied locally to each layer

---

## Compression-Aware Experiment Variants

Three versions of the experiment were run:

1. **Baseline NAS (no compression)**  
   Results from Task 1 using TPESampler

2. **Compression-aware NAS (no post-training)**  
   - Train the model
   - Apply quantisation and pruning
   - Evaluate immediately

3. **Compression-aware NAS (with post-training)**  
   - Train the model
   - Apply quantisation and pruning
   - Fine-tune for one additional epoch
   - Evaluate

---

## Task 2 Results

A second plot was generated with:

- **x-axis**: number of trials
- **y-axis**: best accuracy achieved so far
- **Three curves**:
  - Baseline NAS
  - Compression-aware search without post-training
  - Compression-aware search with post-training

### Observations

- Applying compression without post-training caused a large drop in accuracy
- Post-compression fine-tuning recovered most of the lost performance
- Compression-aware NAS with post-training achieved results close to the baseline

---

## Interpretation

From these results, we can see that:

- Compression changes how a model behaves
- Some architectures are more robust to quantisation and pruning than others
- Fine-tuning after compression is very important
- Including compression during the search leads to better deployment-ready models

---

## Key Takeaways

- TPESampler works better than GridSampler for NAS
- GridSampler cannot handle layer-specific or dynamic parameters
- Compression should not be treated as a separate final step
- Compression-aware NAS produces more robust architectures
- Post-compression training is necessary to achieve good accuracy

---

## Final Insight

Neural architecture search should not be done in isolation.  
If a model is going to be quantised or pruned during deployment, these constraints should be considered **during the search process**, not after.

This tutorial shows how Optuna and MASE can be combined to perform **end-to-end, deployment-aware optimisation**.
