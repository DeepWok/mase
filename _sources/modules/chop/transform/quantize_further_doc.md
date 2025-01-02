# Further Documentation on Quantization

Quantization is a technique to reduce the precision of the weights and activations of a model. MASE supports a wide range of arithmetics with flexible precision settings. This document provides a detailed guide on how to quantize a model with MASE.

The core pass associated with this would be the `quantize_transform_pass` and its test file (usage example) is located in [test_quantize](https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/quantize/test_quantize.py).

Just like other transform passes, this requires a `pass_args` to configure the pass, the following provides an example of the usage:

```python
mg = MaseGraph(model=mlp)
mg, _ = init_metadata_analysis_pass(mg, {})
mg, _ = add_common_metadata_analysis_pass(
		mg, {"dummy_in": dummy_in, "add_value": False}
)
# Sanity check and report
# mg = verify_common_metadata_analysis_pass(mg)
quan_args = {
		"by": "type",
		"default": {"config": {"name": None}},
		"linear": {
				"config": {
						"name": "integer",
						# data
						"data_in_width": 8,
						"data_in_frac_width": 4,
						# weight
						"weight_width": 8,
						"weight_frac_width": 4,
						# bias
						"bias_width": 8,
						"bias_frac_width": 4,
				}
		},
}
mg, _ = quantize_transform_pass(mg, quan_args)
```

The `pass_args` for the `quantize_transform_pass` is a dictionary with the following keys:

- `by`: The quantization strategy, currently `['type', 'name']` are supported.
- `default`: The default quantization configuration for layers that are not defined.
- `{"key" : value}`: The key is the layer name and the value is the quantization configuration for that layer. `key` would be node types if `by` is set to `type` and layer names if `by` is set to `name`. 

The quantization configuration `"config"` is a dictionary with the following keys:

- `name`: The quantization scheme to use.
- `data_in_width`: The bitwidth of the input data.
- `data_in_frac_width`: The fractional bitwidth of the input data.
- `weight_width`: The bitwidth of the weights.
- `weight_frac_width`: The fractional bitwidth of the weights.
- `bias_width`: The bitwidth of the bias.
- `bias_frac_width`: The fractional bitwidth of the bias.

We consider the following quantization ops:

```python
QUANTIZEABLE_OP = (
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "relu",
    "sub",
    "batch_norm2d",
    "layer_norm",
    "group_norm",
    "instance_norm2d",
    "rms_norm",
    "selu",
    "tanh",
    "gelu",
    "softsign",
    "softplus",
)
```

We supported the following quantization schemes that can be set in the `config`:

- `integer`: Integer quantization scheme.
- `fixed`: same as `integer`.
- `lutnet`: [LUTNet](https://arxiv.org/abs/1904.00938) quantization scheme (DEV mode).
- `logicnets`: [LogicNets](https://arxiv.org/abs/2004.03021) quantization scheme (DEV mode).
- `binary`: Binary quantization scheme.
- `binary_residual`: Binary Residual quantization scheme.
- `ternary`: Ternary quantization scheme.
- `minifloat_ieee`: Minifloat quantization scheme, where `exponent_width` and `exponent_bias` are customizable.
- `minifloat_denorm`: Minifloat quantization scheme with denormalization.
- `log`: Logarithmic quantization scheme.
- `block_fp`: Block floating point quantization scheme.
- `block_minifloat`: Block minifloat quantization scheme.
- `block_log`: Block logarithmic quantization scheme.

For blocked-based quantization schemes, see this [paper](https://arxiv.org/abs/2310.05079).