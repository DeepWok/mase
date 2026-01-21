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
