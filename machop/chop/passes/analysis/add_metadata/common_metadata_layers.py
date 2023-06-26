import inspect, torch, math
from chop.passes.utils import vf


# ----------------------------------------------------------
# Placeholder
# ----------------------------------------------------------


def analyse_common_parameters_placeholder(meta, dummy_in):
    """
    The placeholder itself does not contain any information, but can be provided from users.
    """
    var_name = meta.node.target
    if var_name not in dummy_in.keys():
        raise ValueError(f"Cannot find model input {var_name} in the given module")
    arg = dummy_in[var_name]

    meta.parameters["common"]["args"] = {}
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"]["data_out_0"] = {
        "type": "float",
        "precision": [32],
        "size": list(arg.size()),
    }

    return meta


# ----------------------------------------------------------
# Flatten
# ----------------------------------------------------------


def analyse_common_parameters_flatten(meta):
    """
    The placeholder itself does not contain any information, but can be provided from users.
    """
    meta.parameters["common"]["results"] = {}
    start_dim = meta.node.kwargs["start_dim"]
    end_dim = meta.node.kwargs["end_dim"]
    # TODO: Implement a flattening function for the shape
    if start_dim != 1 or end_dim != -1:
        raise NotImplementedError(f"Complex flatten function is not implemented yet...")
    meta.parameters["common"]["results"]["data_out_0"] = meta.parameters["common"][
        "args"
    ]["data_in_0"]
    meta.parameters["common"]["results"]["data_out_0"]["size"] = [
        1,
        math.prod(meta.parameters["common"]["results"]["data_out_0"]["size"]),
    ]

    return meta


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------


def analyse_common_parameters_output(meta):
    """
    Skip as output has no user
    """
    meta.parameters["common"]["results"] = {}
    assert len(meta.node.users.keys()) == 0
    return meta


# ----------------------------------------------------------
# Linear
# ----------------------------------------------------------


def analyse_common_parameters_linear(meta):
    for name, parameter in meta.module.named_parameters():
        meta.parameters["common"]["args"][name] = {
            "type": "float",
            "precision": [32],
            "size": parameter.shape,
        }

    assert hasattr(
        meta.module, "in_features"
    ), f"Linear layer {meta.node.name} does not have in features."
    assert hasattr(
        meta.module, "out_features"
    ), f"Linear layer {meta.node.name} does not have out features."

    assert meta.module.in_features == math.prod(
        meta.parameters["common"]["args"]["data_in_0"]["size"]
    )

    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": (
                1,
                meta.module.out_features,
            ),
        }
    }
    return meta


# ----------------------------------------------------------
# ReLU
# ----------------------------------------------------------
def analyse_common_parameters_relu(meta):
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"]["data_out_0"] = meta.parameters["common"][
        "args"
    ]["data_in_0"]
    return meta
