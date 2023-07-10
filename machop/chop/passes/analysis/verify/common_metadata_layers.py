import math

# ----------------------------------------------------------
# General verification for all kinds of node
# ----------------------------------------------------------


def verify_common_metadata_general(meta):
    """
    Verify general parameters for all the nodes
    """
    # Verify common parameters
    arg_count = len(meta.node.all_input_nodes)
    res_count = len(meta.node.users.keys())
    for i in range(0, arg_count):
        arg_name = f"data_in_{i}"
        assert (
            arg_name in meta.parameters["common"]["args"].keys()
        ), f"Cannot find data_in in common.arg parameters. {meta}"

    for arg, param in meta.parameters["common"]["args"].items():
        ## Valid type
        arg_type = param["type"]
        assert arg_type in meta.known_types, f"Unknown type for {arg} : {arg_type}"
        ## Valid size
        arg_size = param["size"]
        assert arg_size, f"Unknown size for {arg} : {arg_size}"
        ## Data width must be greater than frac width
        if arg_type == "fixed":
            for s in param["size"]:
                assert isinstance(s, int)
            assert isinstance(param["precision"][0], int)
            assert isinstance(param["precision"][1], int)
            assert param["precision"][0] > 0, f"{arg} must have a positive width."
            assert (
                param["precision"][1] >= 0
            ), f"{arg} cannot have a negative frac width."
            assert (
                param["precision"][0] >= param["precision"][1]
            ), f"{arg} must have a width greater than the frac width."
        elif arg_type == "float":
            assert (
                param["precision"][0] == 32 or param["precision"][0] == 64
            ), f"{arg} must have a width of 32 or 64 as float."
        elif arg_type != "NA":
            assert False, f"Unsupported arg type from toml. {param[type]}"

    if arg_count > 0 and res_count > 0:
        assert (
            meta.parameters["common"]["args"]["data_in_0"]["type"]
            == meta.parameters["common"]["results"]["data_out_0"]["type"]
        ), "Input and out data type must match. "

    for result, param in meta.parameters["common"]["results"].items():
        ## Valid type
        result_type = param["type"]
        assert (
            result_type in meta.known_types
        ), f"Unknown type for {result} : {result_type}"
        ## Valid size
        result_size = param["size"]
        assert result_size, f"Unknown size for {result} : {result_size}"
        ## Data width must be greater than frac width
        if result_type == "fixed":
            for s in param["size"]:
                assert isinstance(s, int)
            assert isinstance(param["precision"][0], int)
            assert isinstance(param["precision"][1], int)
            assert param["precision"][0] > 0, f"{result} must have a positive width."
            assert (
                param["precision"][1] >= 0
            ), f"{result} cannot have a negative frac width."
            assert (
                param["precision"][0] >= param["precision"][1]
            ), f"{result} must have a width greater than the frac width."
        elif result_type == "float":
            assert (
                param["precision"][0] == 32 or param["precision"][0] == 64
            ), f"{result} must have a width of 32 or 64 as float."
        else:
            assert False, f"Unsupported result type from toml. {param[type]}"


# ----------------------------------------------------------
# Linear
# ----------------------------------------------------------


def verify_common_metadata_linear(meta):
    if meta.module is not None:
        weight_name = "weight"
        bias_name = "bias"
    else:
        weight_name = "data_in_1"
        bias_name = "data_in_2"

    has_bias = len(meta.node.all_input_nodes) > 2

    # Verify common parameters
    assert (
        meta.parameters["common"]["args"]["data_in_0"]["size"][0]
        == meta.parameters["common"]["args"][weight_name]["size"][1]
    ), "Input row size does not match with the weight col size. {}: in = {}, w = {}".format(
        meta.node.name,
        meta.parameters["common"]["args"]["data_in_0"]["size"],
        meta.parameters["common"]["args"][weight_name]["size"],
    )
    assert (
        meta.parameters["common"]["results"]["data_out_0"]["size"][0]
        == meta.parameters["common"]["args"][weight_name]["size"][0]
    ), "Output row size does not match with the weight row size. {}: out = {}, w = {}".format(
        meta.node.name,
        meta.parameters["common"]["results"]["data_out_0"]["size"],
        meta.parameters["common"]["args"][weight_name]["size"],
    )
    if meta.parameters["common"]["args"]["data_in_0"]["type"] == "fixed":
        # Check the output precision based on the input precision - assume lossless
        # Output width = max(bias_width, data_in_0_width + weight_width + clog2(in_size)) + 1
        # Output frac width = max(bias_frac_width, data_in_0_frac_width + weight_frac_width)
        if has_bias:
            bias_width = meta.parameters["common"]["args"][bias_name]["precision"][0]
            bias_frac_width = meta.parameters["common"]["args"][bias_name]["precision"][
                1
            ]
        else:
            bias_width = 0
            bias_frac_width = 0
        weight_width = meta.parameters["common"]["args"][weight_name]["precision"][0]
        data_in_0_width = meta.parameters["common"]["args"]["data_in_0"]["precision"][0]
        clog2_data_in_0_size = clog2(
            meta.parameters["common"]["args"]["data_in_0"]["size"][1]
        )
        weight_frac_width = meta.parameters["common"]["args"][weight_name]["precision"][
            1
        ]
        data_in_0_frac_width = meta.parameters["common"]["args"]["data_in_0"][
            "precision"
        ][1]
        expected_precision = (
            max(bias_width, weight_width + data_in_0_width + clog2_data_in_0_size) + 1,
            max(bias_frac_width, data_in_0_frac_width + weight_frac_width),
        )
        assert (
            meta.parameters["common"]["results"]["data_out_0"]["precision"]
            == expected_precision
        ), "Output precision does not match with the estimated precision = {}. Expected = {}".format(
            meta.parameters["common"]["results"]["data_out_0"]["precision"],
            expected_precision,
        )


# ----------------------------------------------------------
# ReLU
# ----------------------------------------------------------


def verify_common_metadata_relu(meta):
    # Verify common parameters
    assert (
        meta.parameters["common"]["args"]["data_in_0"]["precision"]
        == meta.parameters["common"]["results"]["data_out_0"]["precision"]
    ), "ReLU has a mismatched input and output pair"
    assert (
        meta.parameters["common"]["args"]["data_in_0"]["type"]
        == meta.parameters["common"]["results"]["data_out_0"]["type"]
    ), "ReLU has a mismatched input and output pair"
    assert (
        meta.parameters["common"]["args"]["data_in_0"]["size"]
        == meta.parameters["common"]["results"]["data_out_0"]["size"]
    ), "ReLU has a mismatched input and output pair"


# ----------------------------------------------------------
# Input is a node that only has output, e.g. Placeholder, Constant
# ----------------------------------------------------------


def verify_common_metadata_input(meta):
    return


# ----------------------------------------------------------
# Flatten
# ----------------------------------------------------------


def verify_common_metadata_flatten(meta):
    # TODO: Implement a flattening function for the shape
    start_dim = meta.node.kwargs["start_dim"]
    end_dim = meta.node.kwargs["end_dim"]
    if start_dim != 1 or end_dim != -1:
        raise NotImplementedError(f"Complex flatten function is not implemented yet...")
    # TODO: Verify the shape is correct
    assert math.prod(
        meta.parameters["common"]["results"]["data_out_0"]["size"]
    ) == math.prod(
        meta.parameters["common"]["args"]["data_in_0"]["size"]
    ), "Output shape does not match with the input shape. out = {}, in = {}".format(
        meta.parameters["common"]["results"]["data_out_0"]["size"],
        meta.parameters["common"]["args"]["data_in_0"]["size"],
    )


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------


def verify_common_metadata_output(meta):
    return
