# General Logbook – Optuna × MASE (Compiler + Search) Notes

## Q: Does Optuna use Bayesian optimisation to choose the next best config?
Optuna is a **hyperparameter optimisation framework**. Whether it behaves like Bayesian optimisation depends on the **sampler**:

- **TPESampler**: Bayesian optimisation–style (sequential model-based).  
  It models the distribution of good vs bad trials and proposes the next configuration expected to improve the objective.
- **RandomSampler**: random exploration.
- **GridSampler**: exhaustive enumeration over a predefined grid.
- Therefore: **Optuna can do Bayesian-style search (TPE), but it is not always Bayesian by default**—it depends on which sampler is used.

Completion sentence (for reports):
> Optuna uses Bayesian optimisation, a family of sequential model-based methods that fit probabilistic models to past trials to select the next configuration that is most likely to improve the objective given the observed results.

---

## Mental Model: Why NAS Fits Perfectly into MASE

### What NAS is in this lab
Neural Architecture Search (NAS) is not about training weights directly; it searches over **structural and hyperparameter choices** that define a model variant, e.g.:
- number of layers
- hidden size
- number of attention heads
- intermediate size
- replacing some Linear layers with Identity (layer choice)

Each trial = a different “program” (a different model instance) that must be trained/evaluated.

### Why it fits MASE naturally
MASE provides an **intermediate representation (IR)** of the model execution:
- PyTorch model → FX trace → MASEGraph (semantic graph + metadata)

So NAS becomes:
> vary the model → compile into IR → run/evaluate → feed results back to search

This is exactly how compiler auto-tuning works.

---

## Optuna + MASE as a “Compiler + Search” Loop

### Mapping the parts
- **Optuna (sampler)** = search strategy (random / grid / Bayesian-style TPE)
- **Model construction** = produces a candidate “program”
- **FX tracing** = captures the executed forward path
- **MASEGraph** = IR with semantic metadata
- **Passes / Pipeline** = compiler optimisation/lowering steps
- **Trainer + Evaluation** = executes the compiled artifact and measures score
- **Objective return value (accuracy)** = cost/fitness function
- **Study history** = data used by the sampler to choose next trial

Closed loop:

trial config → build model → trace (FX) → IR (MASE) → (optional passes) → train/eval → accuracy → sampler updates


---

## Where Quantization Fits (and Why It Matters)

### What quantization changes
Quantization generally does **not** change the high-level topology:
- embeddings → encoder blocks → pooler → classifier

Instead, it changes **numeric representation and arithmetic**, e.g.:
- float32 ops → int8 (or fixed-point) ops + scale/zero-point handling
- reduced precision for weights/activations/bias

Compiler view:
> Quantization is a **lowering step** from high-precision IR to low-precision IR.

### Why MASE/IR is useful for quantization
To quantize safely you need:
- operator identity (what op is this?)
- tensor shapes and dtypes
- producer/consumer relationships (rewiring correctness)
- consistent metadata for downstream passes

FX provides the skeleton of execution.  
MASE adds semantic metadata (shape/type/operator class) to enable principled transforms.

---

## Pruning vs Quantization in the IR View

- **Pruning**: changes parameter sparsity (and potentially removes nodes if rewritten), but often keeps the macro-topology.
- **Quantization**: changes precision/representation; topology usually stays the same.
- Both are expressed as **transform passes** on the MASEGraph and then re-emitted as an executable model.

---

## Why Compression-Aware NAS Is the “Correct” Next Step

### Traditional NAS (what Tutorial 5 starts with)

search architecture (train/eval) → pick best FP32 model → compress afterwards


Problem:
- the best FP32 architecture may be fragile under quantization/pruning.

### Compression-aware NAS (what the task asks for)
Inside each Optuna trial:
1) build model
2) train briefly (warm-up)
3) apply CompressionPipeline (quantize + prune)
4) optionally train a bit more
5) evaluate compressed accuracy
6) return compressed accuracy to Optuna

So the search optimises:
> **post-compression accuracy**, not FP32 accuracy.

This matches compiler auto-tuning:
- the search objective is evaluated **under the deployment constraints** (precision + sparsity).


## Why Graph Topology Often Stays Similar
Even when accuracy changes (different widths, different quant bits, pruning levels), the core forward structure often remains:
- same execution stages (BERT blocks, pooler, classifier)
- differences appear in:
  - layer dimensions / counts
  - numeric formats (quantization)
  - sparsity patterns (pruning)
  - certain modules swapped (e.g. Linear → Identity)

So improved accuracy/efficiency comes mostly from **numerical + structural parameter changes**, not necessarily a completely different macro graph.

---

## Key Takeaway
MASE is not a tool that “recommends” optimisations.  
It is a semantic IR layer that **enables** correct analysis + transformation (quantization, pruning, etc.).  
Optuna provides the search policy that explores configurations and uses measured accuracy to guide exploration (especially with TPE).

Nice, this is a **perfect logbook entry** to have.
Below is a **clean, copy-paste-ready Markdown section** you can drop straight into `labs/notes.md` (or whatever you’re calling your general logbook).

---

# MASE Checkpoints: `.mz` vs `.pt`

## Overview
When exporting a model checkpoint using MASE, two files are produced:

- **`.pt`** — PyTorch checkpoint (weights)
- **`.mz`** — MASE checkpoint (graph + metadata)

Both files are required to fully reconstruct a MASE-managed model. This design mirrors a **compiler-style separation** between *structure* and *data*.

---

## `.pt`: PyTorch Weights Checkpoint

The `.pt` file is a standard PyTorch checkpoint containing:

- Model parameters (weights and biases)
- Buffers (e.g. LayerNorm statistics)

Characteristics:
- Loadable via `torch.load(...)`
- Compatible with vanilla PyTorch
- Contains **no graph structure**
- Contains **no MASE metadata**
- Does **not** record quantization, pruning, or transformation history

Conceptually:
```python
{
  "bert.encoder.layer.0.attention.self.query.weight": Tensor(...),
  "bert.encoder.layer.0.attention.self.query.bias": Tensor(...),
  ...
}
````

This file answers the question:

> *“What are the numerical values of the parameters?”*

---

## `.mz`: MASE Graph + Metadata Checkpoint

The `.mz` file stores the **MASE intermediate representation (IR)**:

* Torch FX graph (forward-pass structure)
* Node-level MASE metadata:

  * operator classification
  * tensor shapes
  * datatypes / precision
  * quantization configuration
  * pruning masks
  * pass history

Conceptually:

```python
{
  "fx_graph": ...,
  "nodes": [
    {
      "name": "bert.encoder.layer.0.attention.self.query",
      "op": "call_module",
      "meta": {
        "mase": {
          "common": {...},
          "hardware": {...},
          "software": {...}
        }
      }
    }
  ]
}
```

This file answers the question:

> *“How is the computation structured after all analysis and transform passes?”*

---

## Why MASE Separates `.pt` and `.mz`

MASE is designed as a **compiler-style system**, not just a training framework.

| Compiler Concept                 | MASE Equivalent             |
| -------------------------------- | --------------------------- |
| Intermediate Representation (IR) | `.mz`                       |
| Object file / parameters         | `.pt`                       |
| Executable                       | Reconstructed PyTorch model |

This separation enables:

* Re-running analysis or transform passes **without retraining**
* Changing quantization or pruning **without re-tracing FX**
* Hardware/software co-design using a shared IR
* Clean separation between *numerical values* and *computational intent*

---

## Why Both Files Are Required

When loading a MASE checkpoint:

1. `.mz` restores the **graph topology and metadata**
2. `.pt` restores the **learned parameters**
3. MASE reconstructs a runnable PyTorch model

If either file is missing:

| Missing File | Result                                                   |
| ------------ | -------------------------------------------------------- |
| `.pt`        | Graph exists, but weights are missing or random          |
| `.mz`        | Weights exist, but graph structure and metadata are lost |
| Both         | Model cannot be reconstructed                            |

---

## Why a Single File Would Be Inferior

Combining graph structure and weights into a single file would:

* Tie metadata changes to weight invalidation
* Complicate versioning and experimentation
* Make hardware lowering and analysis brittle
* Prevent clean reuse of trained parameters across transformations

MASE avoids this by design.

---

## Mental Model

> **`.mz` is the brain** — structure, semantics, optimisation state
> **`.pt` is the memory** — numerical parameters

You need both to fully “run” the model.

---

## Key Takeaway

MASE’s two-file checkpoint system cleanly separates **what the model computes** from **how its parameters are stored**, enabling safe, repeatable graph transformations, compression-aware optimisation, and downstream hardware/software workflows without retracing or retraining.
