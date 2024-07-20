import operator

PYTHON_NATIVE_FUNCTIONS = [
    operator.add,
    operator.mul,
    operator.getitem,
]


def patch_metadata_transform_pass(mg, pass_args: dict = {}):
    """
    Typically, metadata such as precision and type are inferred from each node during the add_common_metadata_analysis_pass.
    However, for call_function nodes where the target is a Python-native function, some metadata is not correctly inferred since
    we avoid overriding Python native functions with mase-specific primitives. Hence, in this pass we manually patch the metadata
    for these nodes according to the requested payloads.
    """

    precision = pass_args.get("precision", "fixed")

    for node in mg.fx_graph.nodes:
        # Update args
        if (
            node.target in PYTHON_NATIVE_FUNCTIONS
            or node.meta["mase"]["common"]["mase_op"] == "df_split"
        ):
            node.meta["mase"]["common"]["args"]["data_in_0"]["type"] = precision
            node.meta["mase"]["common"]["args"]["data_in_0"]["precision"] = [
                pass_args["q_config"]["data_in_width"],
                pass_args["q_config"]["data_in_frac_width"],
            ]
            if "data_in_1" in node.meta["mase"]["common"]["args"]:
                node.meta["mase"]["common"]["args"]["data_in_1"]["type"] = precision
                node.meta["mase"]["common"]["args"]["data_in_1"]["precision"] = [
                    pass_args["q_config"]["data_in_width"],
                    pass_args["q_config"]["data_in_frac_width"],
                ]

        # Update results
        if (
            node.target in PYTHON_NATIVE_FUNCTIONS
            or node.meta["mase"]["common"]["mase_op"] == "df_split"
            or node.op == "placeholder"
            or node.op == "output"
        ):
            node.meta["mase"]["common"]["results"]["data_out_0"]["type"] = precision
            node.meta["mase"]["common"]["results"]["data_out_0"]["precision"] = [
                pass_args["q_config"]["data_out_width"],
                pass_args["q_config"]["data_out_frac_width"],
            ]
            if "data_out_1" in node.meta["mase"]["common"]["results"]:
                node.meta["mase"]["common"]["results"]["data_out_1"]["type"] = precision
                node.meta["mase"]["common"]["results"]["data_out_1"]["precision"] = [
                    pass_args["q_config"]["data_out_width"],
                    pass_args["q_config"]["data_out_frac_width"],
                ]

        # Set one of the args to none according to the select value
        if node.target == operator.getitem:
            select = 0 if node.args[1] == 1 else 1
            node.meta["mase"]["common"]["args"][f"data_in_{select}"] = None

    return mg, {}
