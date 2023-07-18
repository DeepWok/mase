from functools import partial


def entry_to_list(config: dict, entry: str, suffixes: tuple[str]):
    """e.g. [data_in_frac_width, data_in_width]"""
    return list(config[f"{entry}_{suffix}"] for suffix in suffixes)


QUANT_ARITH_TO_SUFFIXES = {
    "integer": ("width", "frac_width"),
    "binary": (
        "width",
        "frac_width",
        "stochastic",
        "bipolar",
    ),  # TODO: stochastic, bipolar flags are operational flag instead of precision.
    "minifloat_ieee": ("width", "exponent_width", "exponent_bias"),
    "minifloat_denorm": ("width", "exponent_width", "exponent_bias"),
    "log": ("width", "exponent_bias"),
    "block_fp": ("width", "exponent_width", "exponent_bias", "block_size"),
    "block_minifloat": ("width", "exponent_width", "exponent_bias_width", "block_size"),
    "block_log": ("width", "exponent_bias_width", "block_size"),
}


# quant_arith_to_list_fn = {
#    <quant_arith>: {
#       <entry>: entry_to_list_<quant_arith>
# }
quant_arith_to_list_fn = {}
for quant_arith, suffixes in QUANT_ARITH_TO_SUFFIXES.items():
    quant_arith_to_list_fn[quant_arith] = partial(entry_to_list, suffixes=suffixes)


def update_arg(node, arg_name, dtype=None, precision=None, size=None):
    if dtype is not None:
        node.meta["mase"].parameters["common"]["args"][arg_name]["type"] = dtype
    if precision is not None:
        node.meta["mase"].parameters["common"]["args"][arg_name][
            "precision"
        ] = precision
    if size is not None:
        node.meta["mase"].parameters["common"]["args"][arg_name]["size"] = size


MASE_OP_TO_INPUT_ENTRIES_AND_ARGS = {
    "add": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
    "bmm": (("data_in", "weight"), ("data_in_0", "data_in_1")),
    "conv1d": (("data_in", "weight", "bias"), ("data_in_0", "weight", "bias")),
    "conv2d": (("data_in", "weight", "bias"), ("data_in_0", "weight", "bias")),
    "matmul": (("data_in", "weight"), ("data_in_0", "data_in_1")),
    "mul": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
    "linear": (("data_in", "weight", "bias"), ("data_in_0", "weight", "bias")),
    "relu": (("data_in",), ("data_in_0",)),
    "sub": (("data_in", "data_in"), ("data_in_0", "data_in_1")),
}


def update_quant_meta_param(node, config: dict, mase_op: str) -> None:
    quant_arith = config["name"]
    assert quant_arith in quant_arith_to_list_fn, f"Unknown quant_arith: {quant_arith}"

    for entry, arg in zip(*MASE_OP_TO_INPUT_ENTRIES_AND_ARGS[mase_op]):
        update_arg(
            node,
            arg_name=arg,
            dtype=quant_arith,
            precision=quant_arith_to_list_fn[quant_arith](config, entry),
        )


def relink_node_meta(node, model):
    node.meta["mase"].node = node
    node.meta["mase"].model = model
