# Lab 1 – Tutorial 3: Bit-Width Sweeps and Accuracy–Precision Trade-offs (MASE Quantisation)

## Objective
Evaluate how **quantisation precision (bit-width)** impacts downstream **classification accuracy**, and observe the practical “knee” where increasing precision yields diminishing returns. This tutorial uses a simple sweep over quantisation widths and records accuracy, then repeats the sweep with an additional training/fine-tuning phase.

---

## Setup
- **Graph**: Same MASEGraph topology as in the earlier screenshot (BERT Tiny sequence classification forward graph).
- **Experiment**: Run the same evaluation pipeline with varying quantisation width.
- **Widths tested**: `4`, `8`, `16`, `32`

Notes on logs:
- Warnings such as:
  - `WARNING  Node finfo not found in loaded metadata.`
  - `WARNING  Node getattr_2 not found in loaded metadata.`
  indicate that some FX nodes present during tracing were not present in the saved/loaded metadata mapping for this run. This did **not** prevent training or evaluation, but suggests metadata caching or node-name alignment was not perfect.
- HuggingFace warning:
  - `FutureWarning: tokenizer is deprecated ... use processing_class instead.`
  is an upstream API change notice and does not affect correctness of results.

---

## Part A: Accuracy Sweep (Evaluation Only)
The first sweep ran quickly (~32–33s per width) and produced:

| Quant Width (bits) | Accuracy |
|---:|---:|
| 4  | 0.67572 |
| 8  | 0.81092 |
| 16 | 0.81508 |
| 32 | 0.81520 |

### Observations
- **4-bit** is significantly worse than the others (large accuracy drop).
- **8-bit** recovers most performance (big jump vs 4-bit).
- **16-bit and 32-bit** are almost identical, indicating a **saturation point**.

### Interpretation
This shows a classic quantisation trade-off:
- At **low precision (4-bit)**, quantisation noise is large enough to distort activations/attention patterns and degrade classification.
- At **moderate precision (8-bit)**, noise is reduced enough for the model to retain most useful signal.
- Beyond **16-bit**, further precision offers minimal benefit because other factors (model capacity, dataset noise, training setup) dominate.

---

## Part B: Accuracy Sweep After Training (1 Epoch)
A second sweep ran with a visible training phase (1 epoch logged) before evaluation.

### Training Loss Snapshots
For **width = 4**, loss decreased then stabilised:
- Step 500: 0.6144  
- Step 1000: 0.5424  
- Step 1500: 0.4969  
- Step 2000: 0.4709  
- Step 2500: 0.4732  
- Step 3000: 0.4808  

For **width = 8/16/32**, loss was lower and very similar across widths:
- Step 500: ~0.415–0.416  
- Step 1000: ~0.401  
- Step 2000: ~0.389  
- Step 3000: ~0.395  

### Post-Training Accuracy
| Quant Width (bits) | Accuracy |
|---:|---:|
| 4  | 0.79256 |
| 8  | 0.83800 |
| 16 | 0.83764 |
| 32 | 0.83776 |

### Observations
- Training dramatically improves **4-bit** accuracy (**0.6757 → 0.7926**).
- **8/16/32** converge to ~0.838 and are effectively the same.
- The gap between **8-bit and 16/32-bit** remains negligible after training.

### Interpretation
Training acts like **adaptation to quantisation**:
- Fine-tuning allows the classifier head and intermediate representations to adjust to quantisation-induced noise.
- The largest gain is at **4-bit**, where noise is strongest and adaptation helps most.
- For **8-bit+**, the model already sits in a regime where quantisation noise is not the dominant failure mode.

---

## Key Result: “Knee” of the Curve
Across both runs, the practical knee is around:

> **8-bit quantisation** — close to full-precision accuracy, with far lower numeric precision requirements than 16/32-bit.

Meanwhile:
- **4-bit** is feasible, but accuracy depends heavily on training/fine-tuning and remains worse than 8-bit.

---

## Why the Graph Topology Stayed the Same
Even though accuracy changed across widths (and training was enabled in the second sweep), the **graph structure** remained the same as the earlier screenshot because:

- Quantisation changes **parameter/activation representation** (values, dtype/scale), not the logical ordering of layers.
- The model still executes:
  `embeddings → encoder blocks → pooler → classifier → output`
- Training introduces gradient computation in the *autograd engine*, but the **forward FX graph** still describes the same forward pass topology.

So accuracy differences come from **numerical effects**, not from changing the sequence of operations.

---

## Key Takeaways
- Quantisation width strongly affects accuracy at very low precision (4-bit).
- 8-bit provides a strong **accuracy–efficiency sweet spot**.
- Fine-tuning helps the most where quantisation is harsh (4-bit).
- FX/MASE graphs are stable across these experiments because the transformation affects **numerics and metadata**, not control flow or module structure.

---

# Lab 1 – Tutorial 4: Structured Pruning and Sparsity–Accuracy Trade-offs (MASE Pruning)

## Objective
Evaluate how **parameter sparsity** affects downstream **classification accuracy**, and compare two pruning strategies — **Random pruning** and **L1-Norm–based pruning** — using the IMDb dataset. The goal is to understand how informed pruning preserves model performance relative to uninformed parameter removal.

---

## Setup
- **Base model**: Best-performing quantised-and-trained model from Tutorial 3 (QAT).
- **Graph**: Same forward MASEGraph topology (BERT Tiny sequence classification).
- **Dataset**: IMDb sentiment classification.
- **Fine-tuning**: 1 epoch after pruning (to adapt surviving parameters).
- **Sparsity range**: `0.1 → 0.9`
- **Pruning methods**:
  - `random`: parameters removed uniformly at random.
  - `l1-norm`: parameters with smallest magnitude removed first.

Each experiment followed the same pipeline:
1. Load the saved MASEGraph checkpoint.
2. Apply pruning at a fixed sparsity.
3. Fine-tune for one epoch.
4. Evaluate accuracy on the test set.

---

## Part A: Random Pruning Results

| Sparsity | Accuracy |
|---:|---:|
| 0.1 | 0.8186 |
| 0.2 | 0.7972 |
| 0.3 | 0.7652 |
| 0.4 | 0.5845 |
| 0.5 | 0.5149 |
| 0.6 | 0.5016 |
| 0.7 | 0.5015 |
| 0.8 | 0.4984 |
| 0.9 | 0.5007 |

### Observations
- Accuracy degrades **rapidly** as sparsity increases.
- Beyond **~40% sparsity**, performance collapses toward **random guessing (~0.5)**.
- Fine-tuning is unable to recover performance once too many critical parameters are removed.

### Interpretation
Random pruning removes parameters **without regard to importance**, which:
- Destroys key attention and feed-forward pathways.
- Prevents the model from preserving learned representations.
- Leads to early catastrophic failure.

This demonstrates that **sparsity alone is not sufficient** — *which* parameters are removed matters.

---

## Part B: L1-Norm Pruning Results

| Sparsity | Accuracy |
|---:|---:|
| 0.1 | 0.8433 |
| 0.2 | 0.8424 |
| 0.3 | 0.8380 |
| 0.4 | 0.8271 |
| 0.5 | 0.8158 |
| 0.6 | 0.8065 |
| 0.7 | 0.7662 |
| 0.8 | 0.6168 |
| 0.9 | 0.5496 |

### Observations
- Accuracy degrades **gradually and smoothly** with increasing sparsity.
- The model remains competitive up to **~60% sparsity**.
- Even at high sparsity (70–80%), L1-norm pruning significantly outperforms random pruning.

### Interpretation
L1-norm pruning removes parameters with the **smallest magnitude**, which are more likely to:
- Be redundant
- Contribute weakly to activations
- Represent noise rather than signal

This preserves the **core computational structure** of the model while reducing parameter count.

---

## Training Efficiency Observations

A notable secondary effect was **training efficiency**:

- **Random pruning**
  - Training time: ~300–530 seconds per run
  - Slower convergence and unstable gradients

- **L1-norm pruning**
  - Training time: ~115–125 seconds per run
  - Faster convergence and more stable loss

This suggests that informed pruning not only preserves accuracy, but also:
- Reduces effective computational complexity
- Improves optimisation stability during fine-tuning

---

## Sparsity–Accuracy Trade-off Summary

When plotted:
- **Random pruning** shows a steep drop-off and early collapse.
- **L1-norm pruning** produces a smooth decay curve, consistently dominating random pruning at all sparsity levels.

The key takeaway:

> **Structured pruning (L1-norm) enables substantial sparsity with minimal accuracy loss, while random pruning fails early.**

---

## Why One Epoch Is Sufficient
The purpose of this experiment is **comparative robustness**, not full retraining.

Using a fixed, small fine-tuning budget:
- Ensures fairness across sparsity levels
- Prevents prolonged retraining from masking pruning damage
- Clearly exposes the structural differences between pruning strategies

Thus, **1 epoch** is sufficient and appropriate for this analysis.

---

## Key Takeaways
- Sparsity alone does not guarantee efficiency — *parameter importance matters*.
- Random pruning leads to early catastrophic accuracy collapse.
- L1-norm pruning preserves performance up to moderate–high sparsity.
- Structured pruning yields both **higher accuracy** and **faster training**.
- MASE enables systematic exploration of these trade-offs via graph-level transformations.

---

[https://mccormickml.com/2024/09/14/qlora-and-4bit-quantization/](https://mccormickml.com/2024/09/14/qlora-and-4bit-quantization/)
