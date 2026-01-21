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


# Lab 0 – Tutorial 2: FX Graph Input Semantics and Forward-Path Sensitivity

## Objective
Understand how **HuggingFace forward-path semantics** interact with **Torch FX symbolic tracing**, and how changing model inputs alters the **captured compute graph topology** in MASE.

This tutorial focuses on how `hf_input_names` controls which branches of `forward()` are executed and therefore traced.

---

## Model Used
- **Model**: `prajjwal1/bert-tiny`
- **API**: `AutoModelForSequenceClassification`
- **Frameworks**: PyTorch + HuggingFace Transformers + Torch FX + MASE

---

## Baseline Configuration
```python
hf_input_names = ["input_ids", "attention_mask", "labels"]
````

### Observed Graph Tail

```
classifier
→ crossentropyloss_0
→ output
```

### Explanation

When `labels` are provided:

* HuggingFace executes **training-mode logic inside `forward()`**
* `CrossEntropyLoss` is computed internally
* FX captures **both inference and loss computation**

This results in an **inference + training hybrid graph**, even though no explicit training loop exists.

---

## Experiment 1: Remove `attention_mask`, Keep `labels`

```python
hf_input_names = ["input_ids", "labels"]
```

### Observed Change

* The `attention_mask` **placeholder node disappears**
* All attention-related masking logic is removed from the graph

### Explanation

When `attention_mask` is omitted **but labels remain**:

* HuggingFace no longer constructs or propagates a mask
* Attention layers operate without masking
* FX traces a **simplified attention path**

This shows that attention masking is **conditionally executed**, not structurally mandatory.

---

## Experiment 2: Remove `labels`, Keep `attention_mask`

```python
hf_input_names = ["input_ids", "attention_mask"]
```

### Observed Change

```
classifier
→ output
```

* `crossentropyloss_0` node is removed
* Graph becomes **pure inference**

### Explanation

Without `labels`:

* HuggingFace does not compute loss
* Forward returns logits only
* FX captures an **inference-only graph**

This is the correct topology for deployment or inference optimisation.

---

## Experiment 3: Remove `attention_mask` and `labels`

```python
hf_input_names = ["input_ids"]
```

### Observed Change

* No external attention mask placeholder
* Attention masking logic may still exist via internally generated defaults
* No loss node

### Explanation

HuggingFace may:

* Generate a default all-ones mask internally
* Preserve internal masking computation

FX traces whatever is **actually executed**, not what is conceptually required.

---

## Key Observations

### 1. FX Is Execution-Path Sensitive

Torch FX captures:

* **Executed Python code**
* Not abstract model structure

Changing inputs changes:

* Which branches execute
* Which nodes appear in the graph

---

### 2. Loss Computation Lives Inside `forward()`

In HuggingFace models:

* Loss is not external
* Supplying `labels` **injects training semantics into the graph**

This is why loss appears as a graph node.

---

### 3. Attention Masking Is Optional, Not Structural

* Attention masking logic only appears if `attention_mask` is used
* Removing it can materially change attention subgraphs

This matters for:

* Hardware emission
* Graph simplification
* Performance modelling

---

## Why This Matters for MASE

MASE operates on **captured compute graphs**, not model definitions.

Therefore:

* Graph topology depends on input configuration
* Training vs inference semantics must be **explicitly controlled**
* Input selection directly impacts:

  * Optimisation passes
  * Hardware mapping
  * Quantisation correctness

---

## Practical Guideline

> Always trace models with the **minimal input set** required for your target workflow.

* Inference → exclude `labels`
* Hardware emit → exclude training-only paths
* Performance analysis → use representative inputs

---

## Key Takeaway

MASE + FX does not reason about *what a model is* — it reasons about **what the model does** for a given execution.

Controlling `hf_input_names` is therefore a **first-class graph design decision**, not a cosmetic change.



