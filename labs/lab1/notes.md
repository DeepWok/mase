# Lab 1 – Tutorial 1: Introduction to MASE IR and FX Graphs

## Objective
Understand how a PyTorch model is imported into MASE via Torch FX, how the resulting compute graph is structured, and how MASE augments FX with semantic metadata to enable principled analysis and transformation passes.

---

## Model Used
- **Model**: `prajjwal1/bert-tiny`
- **Task**: Sequence classification
- **Notes**:
  - The classification head (`classifier.weight`, `classifier.bias`) is randomly initialised.
  - This is expected behaviour when using `AutoModelForSequenceClassification`.
  - The model is suitable for inference-time graph optimisation experiments.

---

## FX Graph Overview
The generated FX graph represents the **entire forward pass** of BERT Tiny, captured using symbolic tracing.

High-level structure:
```

placeholder inputs
→ embeddings
→ encoder layer 0
→ encoder layer 1
→ pooler
→ classifier
→ output

````

Key observations:
- The graph is **linear and acyclic** (no dynamic control flow).
- Encoder layers are **repeated subgraphs**, ideal for automated optimisation.
- Dropout layers are present even though they have no effect during inference.

---

## FX Node Types Observed
Torch FX nodes encode *Python execution semantics*:

- `placeholder`  
  Represents model inputs (e.g. `input_ids`, `attention_mask`).

- `call_module`  
  Invokes `nn.Module.forward()` (e.g. `Linear`, `LayerNorm`, `Dropout`).

- `call_function`  
  Calls free functions (e.g. `torch.matmul`, `operator.add`, `softmax`).

- `call_method`  
  Calls Tensor methods (e.g. `transpose`, `reshape`).

- `output`  
  Represents the return value of `forward()`.

FX preserves **high-level intent**, not low-level kernels.

---

## MASE IR: Why FX Alone Is Not Enough
FX only encodes *how to regenerate Python code*.  
It does **not** describe:
- What operator is being executed (in ML terms)
- Tensor shapes or datatypes
- Hardware or software execution semantics

MASE extends FX by attaching metadata to each node under:
```python
node.meta["mase"] = {
  "common": {},
  "hardware": {},
  "software": {},
}
````

---

## Operator Classification in MASE IR

After running `add_common_metadata_analysis_pass`, each node is annotated with a **MASE operator type**:

* `module_related_func`
  Examples: `Linear`, `LayerNorm`, `ReLU`, `softmax`
  (includes both `nn.Module` and `nn.functional` equivalents)

* `module`
  Modules with no functional counterpart (e.g. `BatchNorm2d`)

* `builtin_func`
  Core tensor operations (e.g. `torch.matmul`, `torch.cat`)

* `placeholder`, `get_attr`, `output`
  Same meaning as in FX

This abstraction allows **hardware and software toolflows to share the same IR**.

---

## Shape Propagation

Using a dummy input (via HuggingFace tokenizer), MASE:

* Executes a traced forward pass
* Records tensor metadata (shape, dtype, stride)
* Stores this under:

```python
node.meta["mase"]["common"]["args"]
node.meta["mase"]["common"]["results"]
```

This enables:

* Safe graph transformations
* Memory / compute analysis
* Parallelisation decisions

---

## Analysis Pass Example: Dropout Counting

A custom analysis pass was written to:

* Traverse all nodes
* Identify dropout layers
* Count their occurrences

Key insight:

> Analysis passes **do not modify graph topology** — they only observe and annotate.

---

## Transform Pass Example: Dropout Removal

A transform pass was implemented to remove all dropout nodes.

Important details:

* `node.replace_all_uses_with(parent_node)` is required
  to rewire downstream consumers.
* Omitting this step causes FX graph validation errors.
* After transformation, the graph remains valid and simpler.

This demonstrates how:

> Transform passes must preserve graph correctness while changing structure.

---

## Why This Matters

The generated MASEGraph:

* Preserves high-level ML semantics
* Is suitable for both software and hardware optimisation
* Enables principled automation (e.g. inference-only simplification)

This tutorial establishes the foundation for:

* Autosharding (software parallelism)
* Hardware lowering (emit Verilog)
* Cost / performance trade-off exploration

---

## Key Takeaway

MASE is not “yet another IR” — it is a **semantic layer on top of FX** that enables safe, automated, and cross-domain optimisation of deep learning systems.

