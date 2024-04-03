from functools import partial


def entry_to_list(config: dict, entry: str, suffixes: tuple[str]):
    """e.g. [data_in_frac_width, data_in_width]"""
    return list(config[f"{entry}_{suffix}"] for suffix in suffixes)


QUANT_ARITH_TO_SUFFIXES = {
    "integer": ("width", "frac_width"),
    "fixed": ("width", "frac_width"),
    "binary": (
        "width",
        "stochastic",
        "bipolar",
    ),  # TODO: stochastic, bipolar flags are operational flag instead of precision.
    "binary_residual": (
        "width",
        "stochastic",
        "bipolar",
    ),  # TODO: stochastic, bipolar flags are operational flag instead of precision.
    "lutnet": ("width", "input_expanded", "k", "binarization_level"),
    "logicnets": ("width", "frac_width"),
    "ternary": ("width", "scaling_factor", "mean", "median", "max"),
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
    # entry and arg corresponding to name in software and hardware mapping
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


def update_result(node, output_name, dtype=None, precision=None, size=None):
    if dtype is not None:
        node.meta["mase"].parameters["common"]["results"][output_name]["type"] = dtype
    if precision is not None:
        node.meta["mase"].parameters["common"]["results"][output_name][
            "precision"
        ] = precision
    if size is not None:
        node.meta["mase"].parameters["common"]["results"][output_name]["size"] = size


MASE_OP_TO_OUTPUT_ENTRIES = {
    # entry and arg corresponding to name in software and hardware mapping
    "add": (("data_out",), ("data_out_0",)),
    "bmm": (("data_out",), ("data_out_0",)),
    "conv1d": (("data_out",), ("data_out_0",)),
    "conv2d": (("data_out",), ("data_out_0",)),
    "matmul": (("data_out",), ("data_out_0",)),
    "mul": (("data_out",), ("data_out_0",)),
    "linear": (("data_out",), ("data_out_0",)),
    "relu": (("data_out",), ("data_out_0",)),
    "sub": (("data_out",), ("data_out_0",)),
}


def arg_exists(node, arg_name) -> bool:
    return arg_name in node.meta["mase"].parameters["common"]["args"]


def update_quant_meta_param(node, config: dict, mase_op: str) -> None:
    quant_arith = config["name"]
    assert quant_arith in quant_arith_to_list_fn, f"Unknown quant_arith: {quant_arith}"
    """
    MASE_OP_TO_INPUT_ENTRIES_AND_ARGS: Give a mapping between config file and mase model
    How it works:
        We find the precision of a certain paramter "e.g data_in" using the precision partial function.

        The precision partial function take a config file and entry "e.g data_in",
        and it will search through all the attributes under this entry based on the quantisation scheme,
        returning a list of precision with the order same as attributes defined in QUANT_ARITH_TO_SUFFIXES

        This precision list is then being mapped to mase data using 'arg'
    """
    for entry, arg in zip(*MASE_OP_TO_INPUT_ENTRIES_AND_ARGS[mase_op]):
        if not arg_exists(node, arg):
            continue
        update_arg(
            node,
            arg_name=arg,
            dtype=quant_arith,
            precision=quant_arith_to_list_fn[quant_arith](config, entry),
        )

    for entry, arg in zip(*MASE_OP_TO_OUTPUT_ENTRIES[mase_op]):
        # Quantise all the output to fixed point. TODO: Make this automatic. Hardware will need change too
        if quant_arith == "binary" or quant_arith == "binary_residual":
            update_result(
                node,
                output_name=arg,
                dtype="binary",
                precision=[32, 0, 1],  # [bitwidth, stochastic, bipolar]
            )


def relink_node_meta(node, model):
    node.meta["mase"].node = node
    node.meta["mase"].model = model
