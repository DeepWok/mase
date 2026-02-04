# Kevin's ADLS Labs Logbook

## Lab 0

### Tutorial 1

BERT - Bidirectional Encoder Representations from Transformers - transformer-based language model from Google (2018)

Characteristics:
- Bidirectional context: Reads entire sequence at once, instead of left-to-right or right-to-left, hecne understanding context from both directions simultaneously
- Pre-trained: On Masked Language Modeling and Next Stence Prediction
- Architecture: Stack of transformer encoder layers with self-attention mechanisms

How does Mase generate a **compute graph**? --> Using Torch FX
Torch FX benefits over other methods:
- Offers high-level IR
- PyTorch native (no need for dedicated runtime, because every operator in the FX graph correlates to a Python object or callable, meaning we can transform and optimize the graph, and regenerate Python code)

How does the MaseTracer class operate?
- Performs symbolic tracing of NN models
- Creates Proxy objects that stand in for real tensor inputs
- Runs a forward pass with these proxies
- Records every operation performed on these proxies
- Builds a graph from the recorded operations

**Mase IR (Intermediate Representation)**

Built on top of Torch FX, but adds semantic meaning about the workload.

Node types in Mase IR:
- `module_related_func`: Functions under `torch.nn.functional` and their `nn.Module` wrappers (e.g., `F.relu` and `nn.ReLU`)
- `module`: `nn.Module` subclasses without a functional counterpart (e.g., `nn.BatchNorm2D`)
- `builtin_func`: `torch` functions not in `nn.functional` (e.g., `torch.cat`, `torch.bmm`)
- `placeholder`, `get_attr`, `output`: Same meaning as in FX

Metadata categories: `common`, `hardware`, `software`

**Pass System**

Passes iterate over graph nodes to analyze or transform them.

Two categories:
- Analysis passes: Extract info, annotate nodes, generate payloads for subsequent passes
- Transform passes: Change graph topology (insert, remove, replace nodes)

Pass signature:
```python
def my_pass(mg, pass_args={}):
    pass_outputs = {}
    for node in mg.fx_graph.nodes:
        # ... do stuff
    return mg, pass_outputs
```

**Raising FX to Mase IR**

Two required passes:
1. `init_metadata_analysis_pass`: Initializes empty `MaseMetadata` dict on each node
2. `add_common_metadata_analysis_pass`: Populates metadata via:
   - Operator inference: Determines Mase operator type for each node
   - Shape propagation: Runs forward pass with dummy input, records tensor shapes/dtypes

**Writing Passes (Examples)**

Analysis pass (count dropout layers):
- Iterate nodes, check `node.op == "call_module"` and `"dropout" in node.target`

Transform pass (remove dropout):
- Before erasing a node, call `node.replace_all_uses_with(parent_node)` to rewire dependencies
- Then call `mg.fx_graph.erase_node(node)`

**Exporting MaseGraph**

```python
mg.export("path/to/checkpoint")  # Saves .pt (GraphModule) + .mz (metadata)
mg = MaseGraph.from_checkpoint("path/to/checkpoint")  # Reload later
```

---

### Tutorial 2: LoRA Finetuning

**Sentiment Analysis with IMDb Dataset**

IMDb dataset: 50k movie reviews labeled "positive" or "negative", commonly used for NLP sentiment analysis.

Loading dataset in Mase:
```python
from chop.tools import get_tokenized_dataset
dataset, tokenizer = get_tokenized_dataset(
    dataset="imdb",
    checkpoint="bert-base-uncased",
    return_tokenizer=True,
)
```

**Generating MaseGraph with Custom Arguments**

BERT's forward function has many optional arguments. By default, MaseGraph only uses `input_ids`. You can specify additional inputs using `hf_input_names`:
```python
mg = MaseGraph(
    model,
    hf_input_names=["input_ids", "attention_mask", "labels"],
)
```
- Including `labels` adds `nn.CrossEntropyLoss` at the end of the model to calculate loss directly

**Full Supervised Finetuning (SFT)**

Inspect trainable parameters using:
```python
from chop.passes.module import report_trainable_parameters_analysis_pass
_, _ = report_trainable_parameters_analysis_pass(mg.model)
```

Most parameters are in the Embedding layer - freeze these since they don't need training:
```python
for param in mg.model.bert.embeddings.parameters():
    param.requires_grad = False
```

Training uses HuggingFace Trainer via Mase utility:
```python
from chop.tools import get_trainer
trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)
trainer.train()
eval_results = trainer.evaluate()
```

**Parameter Efficient Finetuning (PEFT) with LoRA**

LoRA (Low-Rank Adaptation) - proposed by Microsoft in 2021.

Standard linear layer equation:
```
y = X @ W + b
```

LoRA modification:
```
y = X @ (W + A @ B) + b
```
where A and B are low-rank matrices. Freeze W, only train A and B.

Benefits:
- Reduces trainable parameters by ~4.5x
- Similar accuracy to full finetuning
- Much lower memory requirement for training

Inject LoRA adapters:
```python
mg, _ = passes.insert_lora_adapter_transform_pass(
    mg,
    pass_args={
        "rank": 6,      # rank of A and B matrices
        "alpha": 1.0,   # scaling factor
        "dropout": 0.5,
    },
)
```

After training, fuse LoRA weights for inference optimization:
```python
mg, _ = passes.fuse_lora_weights_transform_pass(mg)
```
This merges the A @ B product into the original weights matrix W, reducing kernel invocations at inference time.

---

### Tutorial 3: Quantization-Aware Training (QAT)

**Why Quantization?**

Reduces model precision (e.g., FP32 → INT8) to:
- Decrease memory footprint
- Speed up inference
- Enable deployment on edge devices

**Post-Training Quantization (PTQ)**

Simply quantize the model after training without any further adjustments.
- Fast and simple
- May cause significant accuracy drop

**Quantization Configuration**

```python
quantization_config = {
    "by": "type",  # Quantize by operator type (can also be "name" or "regex")
    "default": {
        "config": {"name": None}  # Don't quantize by default
    },
    "linear": {
        "config": {
            "name": "integer",
            # Input activations
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # Weights
            "weight_width": 8,
            "weight_frac_width": 4,
            # Biases
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)
```

- `width`: Total number of bits
- `frac_width`: Number of fractional bits (for fixed-point representation)

**Quantization-Aware Training (QAT)**

Include the quantized model back in the training loop:
```python
trainer.train()  # Trains quantized model
eval_results = trainer.evaluate()
```

The model learns to compensate for quantization errors during training. QAT can match or even exceed full-precision accuracy with much lower memory requirement.

---

### Tutorial 4: Pruning

**What is Pruning?**

Removing unnecessary parameters (weights/connections) or structural components (neurons/filters/layers) from neural networks.

**Benefits**:
- Reduce model size (fewer parameters = less storage)
- Decrease inference time (fewer computations)
- Improve generalization (can prevent overfitting)
- Energy efficiency (important for edge/mobile devices)

**Types of Pruning**:

1. **Structured Pruning**: Remove entire structures
   - Channels, filters, or entire layers
   - Maintains regular tensor shapes
   - Hardware-friendly

2. **Unstructured Pruning**: Remove individual weights
   - Regardless of location in tensor
   - Creates sparse weight matrices
   - Higher compression potential but may need sparse hardware support

**Pruning Configuration**

```python
pruning_config = {
    "weight": {
        "sparsity": 0.5,      # Proportion to prune (0-1), 0.5 = 50% zeros
        "method": "l1-norm",  # or "random"
        "scope": "local",     # or "global"
    },
    "activation": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
}

mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
```

**Parameters**:
- `sparsity`: Fraction of elements to set to zero
- `method`:
  - `random`: Randomly select elements to prune
  - `l1-norm`: Prune elements with smallest absolute values
- `scope`:
  - `local`: Consider each tensor individually when computing threshold
  - `global`: Consider all tensors together when computing threshold

**Post-Pruning Finetuning**

Pruning typically causes accuracy drop (e.g., 84% → 55%). Run additional finetuning epochs to allow model to adapt to pruning mask:
```python
trainer = get_trainer(..., num_train_epochs=5)
trainer.train()
# Accuracy recovers (e.g., 55% → 83%)
```

---

### Tutorial 5: Neural Architecture Search (NAS) with Optuna

**What is NAS?**

Automatically search for optimal model architecture instead of manual design.

**Integration with Optuna**

Optuna is a popular hyperparameter optimization framework. NAS workflow:

**Step 1: Define Search Space**
```python
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [nn.Linear, Identity],  # Can skip layers
}
```

**Step 2: Write Model Constructor**
```python
def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)

    # Sample parameters using Optuna
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])

    trial_model = AutoModelForSequenceClassification.from_config(config)

    # Optionally replace layers (e.g., Linear → Identity to skip)
    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear):
            new_layer_cls = trial.suggest_categorical(f"{name}_type", ...)
            # Replace layer if needed

    return trial_model
```

**Step 3: Define Objective Function**
```python
def objective(trial):
    model = construct_model(trial)
    trainer = get_trainer(model=model, ...)
    trainer.train()
    eval_results = trainer.evaluate()
    trial.set_user_attr("model", model)  # Store for later retrieval
    return eval_results["eval_accuracy"]
```

**Step 4: Launch Search**
```python
from optuna.samplers import RandomSampler, GridSampler, TPESampler

sampler = RandomSampler()  # or TPESampler() for smarter sampling

study = optuna.create_study(
    direction="maximize",
    study_name="bert-nas-study",
    sampler=sampler,
)

study.optimize(objective, n_trials=100, timeout=60*60*24)

# Retrieve best model
best_model = study.best_trial.user_attrs["model"]
```

**Optuna Samplers**:
- `GridSampler`: Exhaustive search through all combinations
- `RandomSampler`: Random sampling from search space
- `TPESampler`: Tree-structured Parzen Estimator - learns from previous trials

**CompressionPipeline**

Combine multiple optimization passes in one:
```python
from chop.pipelines import CompressionPipeline

pipe = CompressionPipeline()
mg, _ = pipe(
    mg,
    pass_args={
        "quantize_transform_pass": quantization_config,
        "prune_transform_pass": pruning_config,
    },
)
```

---

### Tutorial 6: Mixed Precision Quantization Search

**What is Mixed Precision?**

Unlike uniform quantization (same precision everywhere), mixed precision allows:
- Different precisions for weights, activations, biases
- Different precisions for different layers

**Why Mixed Precision?**

Some layers are more sensitive to quantization than others. Using higher precision only where needed allows:
- Better accuracy than uniform low precision
- Better efficiency than uniform high precision

**Observing Precision Impact**

```python
# High precision config (16-bit)
quantization_config_high = {..., "data_in_width": 16, "weight_width": 16, ...}

# Low precision config (4-bit)
quantization_config_low = {..., "data_in_width": 4, "weight_width": 4, ...}

# Results comparison:
# Original Accuracy: 82.18%
# High (16-bit): 82.18%  - No accuracy loss
# Low (4-bit): 50%       - Significant accuracy loss (random guessing)
```

**Search Space for Mixed Precision**
```python
search_space = {
    # (width, frac_width) pairs
    "data_in": [(4,2), (4,3), (6,2), (6,4), (8,2), (8,4), (8,6),
                (16,2), (16,4), (16,6), (16,8), (16,10), (16,12)],
    "weight": [...],  # Same options
    "bias": [...],    # Same options
}
```

**Model Constructor for Mixed Precision Search**
```python
def construct_model(trial):
    config = copy.deepcopy(quantization_config)

    for param in ["data_in", "weight", "bias"]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        width, frac_width = search_space[param][chosen_idx]
        config["linear"]["config"][f"{param}_width"] = width
        config["linear"]["config"][f"{param}_frac_width"] = frac_width

    mg_q, _ = passes.quantize_transform_pass(mg, pass_args=config)
    return mg_q
```

Uses same Optuna workflow as NAS to find optimal precision configuration.

---

### Tutorial 7: Distributed Deployment

**Goal**: Deploy models for inference on distributed clusters

*(Tutorial content not yet available)*

---

### Tutorial 8: Emit Verilog (FPGA Accelerator)

**Goal**: Auto-generate FPGA accelerator for transformer models

*(Tutorial content not yet available)*

---

### Tutorial 9: Kernel Fusion

**What is a GPU Kernel?**

A kernel (in GPU context) is a function that runs in parallel across the GPU fabric. Different from OS kernel concept.

A custom kernel is written by developers (not provided by GPU vendor) to optimize specific operations.

**What is Kernel Fusion?**

Combining instructions from multiple kernels into a single kernel.

**Benefits**:
1. Fewer kernel launches (launches have overhead)
2. Reduced memory accesses (data stays in fast memory)
3. Fewer intermediate results stored in slow memory (HBM)

**Example: Naive Softmax Memory Operations**
```python
def torch_naive_softmax(x):
    x_max = x.max(dim=1)[0]           # Read MN, Write M
    z = x - x_max[:, None]             # Read MN+M, Write MN
    numerator = torch.exp(z)           # Read MN, Write MN
    denominator = numerator.sum(dim=1) # Read MN, Write M
    ret = numerator / denominator[:, None]  # Read MN+M, Write MN
    # Total: Read 5MN+2M, Write 3MN+2M
    return ret
```

**What is Triton?**

Triton sits between Python's accessibility and CUDA's fine-grained control:
- Python-like syntax
- Control over HBM (High-Bandwidth Memory) operations
- Control over parallelism levels
- No need to write raw CUDA

**Fused Softmax in Triton**

Key optimizations:
- Load entire row into SRAM once
- Perform all operations (max, subtract, exp, sum, divide) in SRAM
- Write result back to DRAM once
- Uses `tl.load`, `tl.store`, `tl.exp`, `tl.max`, `tl.sum` primitives

Important concepts:
- `BLOCK_SIZE`: Power of 2 >= number of columns
- `num_warps`: Threads per row (for parallelism)
- `num_stages`: Software pipelining stages
- Masking for handling when BLOCK_SIZE > actual columns

---

### Tutorial 10: Mixed Precision Search (Part 2)

**Building on Tutorial 6**

This tutorial demonstrates more advanced mixed precision search techniques.

**Comparing Precision Configurations**

Same model with different precision setups:
```python
# Apply high precision
mg_high, _ = passes.quantize_transform_pass(mg, pass_args=config_high)

# Apply low precision
mg_low, _ = passes.quantize_transform_pass(mg, pass_args=config_low)

# Compare accuracies
accuracy_high = get_accuracy(mg_high, dataset, tokenizer)  # ~82%
accuracy_low = get_accuracy(mg_low, dataset, tokenizer)    # ~50%
```

**Key Insight**: Precision has dramatic impact on accuracy. Finding the right balance through search is crucial for efficient deployment.

**Automated Search Process**

Same Optuna workflow:
1. Define search space with (width, frac_width) pairs
2. Model constructor samples precision config per trial
3. Objective function evaluates accuracy
4. Optuna sampler explores search space intelligently

The search finds configurations that maintain high accuracy while using lower precision where possible.

