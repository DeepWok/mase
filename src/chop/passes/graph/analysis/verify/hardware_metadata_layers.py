# ----------------------------------------------------------
# General verification for all kinds of node
# ----------------------------------------------------------


def verify_hardware_metadata_general(meta):
    """
    Verify general parameters for all the metas
    """
    if meta.parameters["hardware"]["is_implicit"]:
        return

    # Verify hardware parameters
    for name, param in meta.parameters["hardware"]["interface_parameters"].items():
        storage_param = param["storage"]
        assert (
            storage_param in meta.known_storage
        ), f"Invalid parameter storage = {storage_param} for {name}. {meta.meta}"
    toolchain = meta.parameters["hardware"]["toolchain"]
    assert (
        toolchain in meta.known_toolchain
    ), f"Invalid parameter toolchain = {TARGET}. {meta.meta}"

    for name, param in meta.parameters["hardware"]["verilog_parameters"].items():
        assert isinstance(param, int), f"{name} must be int type. {meta.meta}"


# ----------------------------------------------------------
# Linear
# ----------------------------------------------------------


def verify_hardware_metadata_linear(meta):
    data_in_param = meta.parameters["common"]["args"]["data_in"]
    if data_in_param["type"] == "fixed":
        assert isinstance(data_in_param["precision"][0], int)
        weight_param = meta.parameters["common"]["args"]["weight"]
        bias_param = meta.parameters["common"]["args"]["bias"]
        assert meta.parameters["hardware"]["verilog_parameters"]["IN_SIZE"] > 0
        assert meta.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"] > 0
        assert (
            meta.parameters["hardware"]["verilog_parameters"]["IN_DEPTH"]
            * meta.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
            == data_in_param["size"][1]
        )
        assert meta.parameters["hardware"]["verilog_parameters"]["PARALLELISM"] > 0
        assert meta.parameters["hardware"]["verilog_parameters"]["HAS_BIAS"] in [
            0,
            1,
        ], f"Invalid parameter HAS_BIAS = {HAS_BIAS}. {meta.node}"
        # WEIGHT_SIZE == IN_SIZE * PARALLELISM
        assert (
            meta.parameters["hardware"]["verilog_parameters"]["WEIGHT_SIZE"]
            == meta.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
            * meta.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
        )
        # OUT_SIZE == PARALLELISM
        assert (
            meta.parameters["hardware"]["verilog_parameters"]["OUT_SIZE"]
            == meta.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
        )
        # BIAS_SIZE == PARALLELISM
        assert (
            meta.parameters["hardware"]["verilog_parameters"]["BIAS_SIZE"]
            == meta.parameters["hardware"]["verilog_parameters"]["PARALLELISM"]
        )
    else:
        assert False, "Unsupported arg type from toml. Only fixed is supported."


# ----------------------------------------------------------
# ReLU
# ----------------------------------------------------------


def verify_hardware_metadata_relu(meta):
    # Verify common parameters
    assert (
        meta.parameters["common"]["args"]["data_in"]
        == meta.parameters["common"]["results"]["data_out"]
    ), "ReLU has a mismatched input and output pair"

    # Verify hardware parameters
    data_in_param = meta.parameters["common"]["args"]["data_in"]
    if data_in_param["type"] == "fixed":
        assert meta.parameters["hardware"]["verilog_parameters"]["IN_SIZE"] > 0
        assert (
            meta.parameters["hardware"]["verilog_parameters"]["IN_SIZE"]
            == meta.parameters["hardware"]["verilog_parameters"]["OUT_SIZE"]
        )
    else:
        assert False, "Unsupported arg type from toml. Only fixed is supported."
