# Optical Neural Network (ONN) Transform API

This module provides tools for transforming PyTorch neural networks to simulate optical computing behavior, based on the [Optical Transformers paper](https://arxiv.org/abs/2302.10360).

## Installation

```bash
pip install mase-triton
```

## Quick Start

```python
from chop.passes.module.transforms.onn.transform import (
    OtTransformConfig,
    optical_transformer_module_transform_pass,
)

# Create configuration
config = OtTransformConfig.create_default()

# Transform a model
pass_args = {
    "by": "regex_name",
    r"model\.layers\.\d+\.self_attn": config,
    r"model\.layers\.\d+\.mlp\..*_proj": config,
}
model = optical_transformer_module_transform_pass(model, pass_args)
```

## API Reference

### `OtTransformConfig`

Configuration dictionary for optical transform parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q_levels` | int | 256 | Number of quantization levels ($2^n$ for n-bit) |
| `q_lut_min` | float | 0.020040 | Minimum LUT value for quantization |
| `q_smooth_factor` | float | 0.9 | Smoothing factor for statistics updates |
| `q_init_seed` | int | 0 | Random seed for Triton kernels |
| `q_bypass` | bool | False | Bypass optical quantization if True |

```python
# Create default config
config = OtTransformConfig.create_default()

# Customize
config["q_levels"] = 512  # 9-bit quantization
config["q_smooth_factor"] = 0.1
```

### `optical_transformer_module_transform_pass`

Transform supported modules in a network to their optical equivalents.

```python
optical_transformer_module_transform_pass(network, pass_args) -> torch.nn.Module
```

**Parameters:**
- `network`: The PyTorch model to transform
- `pass_args`: Configuration dictionary with:
  - `by`: Matching mode - `"name"` (exact) or `"regex_name"` (regex pattern)
  - Layer patterns mapped to `OtTransformConfig` dicts
  - `default`: Optional fallback config

**Supported Transformations:**

| Original Module | Optical Equivalent |
|-----------------|-------------------|
| `torch.nn.Linear` | `OtLinear` |
| `LlamaAttention` | `OtLlamaAttention` |

### `OtLinear`

Optical equivalent of `torch.nn.Linear` with quantized matrix multiplication.

```python
from chop.passes.module.transforms.onn.transform import OtLinear

# Convert from existing linear layer
linear_onn = OtLinear.from_linear(linear, **config)
```

### `OtLlamaAttention`

Optical equivalent of HuggingFace's `LlamaAttention` with quantized scaled dot-product attention.

```python
from chop.passes.module.transforms.onn.transform import OtLlamaAttention

# Convert from existing attention layer
attn_onn = OtLlamaAttention.from_pretrained(attn, **config)
```

## Important Notes

### Quantization Statistics Warmup

Optical layers require calibration before use. Run a few forward passes in **training mode** first:

```python
model.train()
with torch.no_grad():
    for batch in warmup_batches:
        _ = model(**batch)
```

Without warmup, statistics are `[inf, -inf]` and outputs will be NaN.

### Training vs Evaluation Mode

- **Training mode** (`model.train()`): Statistics are updated with each forward pass
- **Evaluation mode** (`model.eval()`): Statistics are frozen

### Attention Implementation

When loading HuggingFace models, use eager attention for compatibility:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
)
```


## Source Code

- Transform pass: `src/chop/passes/module/transforms/onn/transform.py`
- Linear layer: `src/chop/passes/module/transforms/onn/layers/linear.py`
- Attention layer: `src/chop/passes/module/transforms/onn/layers/attn.py`

## References

- [Optical Transformers: End-to-end Optical Training of Transformer Models](https://arxiv.org/abs/2302.10360)
