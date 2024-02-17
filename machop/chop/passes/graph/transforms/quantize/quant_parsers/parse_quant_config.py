from functools import partial

from .utils import cp_multi_values, has_multi_keys

""" QUANT_ARITH_ENTRIES
A mapping from (quantization arithmetic name) to (a mapping from (operand name) to (operand quantization spec name))

Example

A fixed point quantized value is defined by (width, frac_width), thus the mapping is defined as follows:
```python
"fixed": {
    "weight_entries": ("weight_width", "weight_frac_width"),
    "data_in_entries": ("data_in_width", "data_in_frac_width"),
    "bias_entries": ("bias_width", "bias_frac_width"),
},
```
"""
QUANT_ARITH_ENTRIES = {
    # <arith_name> : {<operand_name> : (<operand_quantization_spec_name>,)}
    "integer": {
        "weight_entries": ("weight_width", "weight_frac_width"),
        "data_in_entries": ("data_in_width", "data_in_frac_width"),
        "bias_entries": ("bias_width", "bias_frac_width"),
    },
    "fixed": {
        "weight_entries": ("weight_width", "weight_frac_width"),
        "data_in_entries": ("data_in_width", "data_in_frac_width"),
        "bias_entries": ("bias_width", "bias_frac_width"),
    },
    "lutnet": {
        "weight_entries": (
            "weight_width",
            "weight_frac_width",
            "weight_binarization_level",
            "weight_input_expanded",
            "weight_k",
            "weight_in_dim",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_frac_width",
            "data_in_binarization_level",  # binarization_level (int): which level of binarization is applied, "binarized_weight" is only weights binarized others is no binarization
            "data_in_input_expanded",  # input_expanded (bool): If set to True, means all LUT's inputs are considered during calculations , else only the first input will considered and the remaining will be masked.
            "data_in_k",  # k entries of a LUT
            "data_in_levels",  # data_in_levels (int): number of residual levels to use in lutnet
            "data_in_dim",  # data input dimension (this is needed by convolution)
        ),
        "bias_entries": (
            "bias_width",
            "bias_frac_width",
            "bias_binarization_level",
            "bias_input_expanded",
            "bias_k",
            "bias_in_dim",
        ),
    },
    "logicnets": {
        "weight_entries": (  # TODO: change update_node_meta.py to take optional argument so this can be removed
            "weight_width",
            "weight_frac_width",
        ),
        "bias_entries": (
            "bias_width",
            "bias_frac_width",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_frac_width",
        ),
        "data_out_entries": (
            "data_out_width",
            "data_out_frac_width",
        ),
        "additional_layers_entries": {
            "additional_layers_inputs",
            "additional_layers_outputs",
        },
    },
    "binary": {
        "weight_entries": (
            "weight_width",
            "weight_stochastic",
            "weight_bipolar",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_stochastic",
            "data_in_bipolar",
        ),
        "bias_entries": (
            "bias_width",
            "bias_stochastic",
            "bias_bipolar",
        ),
    },
    "binary_residual": {
        "weight_entries": (
            "weight_width",
            "weight_stochastic",
            "weight_bipolar",
            "binary_training",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_stochastic",
            "data_in_bipolar",
            "data_in_residual_sign",
            "data_in_levels",  # data_in_levels (int): number of residual levels to use in lutnet
        ),
        "bias_entries": (
            "bias_width",
            "bias_stochastic",
            "bias_bipolar",
        ),
    },
    "binary_residual": {
        "weight_entries": (
            "weight_width",
            "weight_stochastic",
            "weight_bipolar",
            "binary_training",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_stochastic",
            "data_in_bipolar",
            "data_in_residual_sign",
            "data_in_levels",  # data_in_levels (int): number of residual levels to use in lutnet
        ),
        "bias_entries": (
            "bias_width",
            "bias_stochastic",
            "bias_bipolar",
        ),
    },
    "ternary": {
        "weight_entries": (
            "weight_width",
            "weight_scaling_factor",
            "weight_mean",
            "weight_median",
            "weight_max",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_scaling_factor",
            "data_in_mean",
            "data_in_median",
            "data_in_max",
        ),
        "bias_entries": (
            "bias_width",
            "bias_scaling_factor",
            "bias_mean",
            "bias_max",
            "bias_median",
        ),
    },
    "minifloat_ieee": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
        ),
        "bias_entries": ("bias_width", "bias_exponent_width", "bias_exponent_bias"),
    },
    "minifloat_denorm": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
        ),
        "bias_entries": ("bias_width", "bias_exponent_width", "bias_exponent_bias"),
    },
    "log": {
        "weight_entries": ("weight_width", "weight_exponent_bias"),
        "data_in_entries": ("data_in_width", "data_in_exponent_bias"),
        "bias_entries": ("bias_width", "bias_exponent_bias"),
    },
    "block_fp": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_width",
            "bias_exponent_bias",
            "bias_block_size",
        ),
    },
    "block_minifloat": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias_width",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias_width",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_width",
            "bias_exponent_bias_width",
            "bias_block_size",
        ),
    },
    "block_log": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_bias_width",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_bias_width",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_bias_width",
            "bias_block_size",
        ),
    },
}


""" cp_<entry_name> functions
A collection of functions to copy values from a src config to a parsed config.
"""


def cp_name(config: dict, p_config: dict, entries=None, strict: bool = True):
    cp_multi_values(config, p_config, ("name",), strict=strict)


def cp_bypass(config: dict, p_config: dict, entries=None, strict: bool = True):
    cp_multi_values(config, p_config, ("bypass",), strict=strict)


def cp_weight_entries(config: dict, p_config: dict, entries: dict, strict: bool = True):
    cp_multi_values(config, p_config, entries["weight_entries"], strict=strict)


def cp_data_in_entries(
    config: dict, p_config: dict, entries: dict, strict: bool = True
):
    cp_multi_values(config, p_config, entries["data_in_entries"], strict=strict)


def cp_data_out_entries(config: dict, p_config: dict, entries: dict):
    cp_multi_values(config, p_config, entries["data_out_entries"])


def cp_bias_entries(config: dict, p_config: dict, entries: dict, strict: bool = True):
    cp_multi_values(config, p_config, entries["bias_entries"], strict=strict)


def cp_weight_entries_to_bias(
    config: dict, p_config: dict, entries: dict, strict: bool = True
):
    if has_multi_keys(config, entries["bias_entries"]):
        cp_multi_values(config, p_config, entries["bias_entries"], strict=strict)
    else:
        cp_multi_values(
            config,
            p_config,
            entries["weight_entries"],
            entries["bias_entries"],
            strict=strict,
        )


def cp_layer_entries(config: dict, p_config: dict, entries: dict, strict: bool = True):
    cp_multi_values(config, p_config, entries["additional_layers_entries"])


def cp_data_out_entries(
    config: dict, p_config: dict, entries: dict, strict: bool = True
):
    cp_multi_values(config, p_config, entries["data_out_entries"], strict=strict)


"""QUANT_ARITH_TO_CP_FN
a map from quant_arith to a collection of functions where each function copies a specific quant_arith_spec from a src config to a parsed config.

<quant_arith>: {
   "name": cp_name_function_<quant_arith>,
   "weight_entries": cp_weight_entries_function_<quant_arith>,
   "data_in_entries": cp_data_in_entries_function_<quant_arith>,
   "bias_entries": cp_bias_entries_function_<quant_arith>,
   "weight_entries_to_bias": cp_weight_entries_to_bias_function_<quant_arith>
}
"""
QUANT_ARITH_TO_CP_FN = {}


for quant_arith, entries in QUANT_ARITH_ENTRIES.items():
    QUANT_ARITH_TO_CP_FN[quant_arith] = {
        "name": partial(cp_name, entries=entries),
        "bypass": partial(cp_bypass, entries=entries),
        "weight_entries": partial(cp_weight_entries, entries=entries),
        "data_in_entries": partial(cp_data_in_entries, entries=entries),
        "bias_entries": partial(cp_bias_entries, entries=entries),
        "data_out_entries": partial(cp_data_out_entries, entries=entries),
        "weight_entries_to_bias": partial(cp_weight_entries_to_bias, entries=entries),
        "additional_layers_entries": partial(cp_layer_entries, entries=entries),
    }

""" MASE_OP_TO_ENTRIES
a map from mase_op to a collection of required and optional entries.
"""
MASE_OP_TO_ENTRIES = {
    # <op_name> : (<required_entries>, <optional_entries>)
    "add": (("name", "data_in_entries"), ("bypass",)),
    "bmm": (("name", "data_in_entries", "weight_entries"), ("bypass",)),
    "conv1d": (
        ("name", "data_in_entries", "weight_entries"),
        ("bias_entries", "bypass"),
    ),
    "conv2d": (
        ("name", "data_in_entries", "weight_entries"),
        ("bias_entries", "bypass", "data_out_entries"),
    ),
    "matmul": (("name", "data_in_entries", "weight_entries"), ("bypass",)),
    "mul": (("name", "data_in_entries"), ("bypass",)),
    "linear": (
        ("name", "data_in_entries", "weight_entries"),
        ("bias_entries", "bypass", "data_out_entries", "additional_layers_entries"),
    ),
    "relu": (("name", "data_in_entries"), ("bypass",)),
    "sub": (("name", "data_in_entries"), ("bypass",)),
    "rotary_positional_encoding": (
        ("name", "data_in_entries"),
        ("bypass",),
    ),  # RoPE of Llama
}


def optional_operand_entry_exists(config: dict, entry_name: str) -> bool:
    entry_name = entry_name.removesuffix("_entries")
    for key in config.keys():
        if key.startswith(entry_name):
            return True
    return False


def parse_node_config(config: dict, mase_op: str, strict: bool = True) -> dict:
    """
    Parse a node config from a MASE op config.

    Args:
        - `strict` (bool) allows missing node config entries if False, e.g.,
        a missing `bias_frac_width` in linear node config
    """
    assert mase_op in MASE_OP_TO_ENTRIES, f"Unknown mase op: {mase_op}"
    if config.get("bypass", False):
        return config
    op_entries, op_optional_entries = MASE_OP_TO_ENTRIES[mase_op]
    assert isinstance(op_entries, tuple), f"op_entries must be a tuple: {op_entries}"
    assert isinstance(
        op_optional_entries, tuple
    ), f"op_optional_entries must be a tuple: {op_optional_entries}"
    p_config = {}
    for entry in op_entries:
        entry_cp_fn = QUANT_ARITH_TO_CP_FN[config["name"]][entry]
        entry_cp_fn(config, p_config, strict=strict)
    for entry in op_optional_entries:
        if optional_operand_entry_exists(config, entry):
            entry_cp_fn = QUANT_ARITH_TO_CP_FN[config["name"]][entry]
            entry_cp_fn(config, p_config, strict=strict)
    return p_config
