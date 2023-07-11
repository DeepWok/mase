import inspect
import math

import torch
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
# View
# ----------------------------------------------------------


def analyse_common_parameters_view(meta):
    """
    Memory transformation.
    """
    meta.parameters["common"]["results"] = {}
    if meta.node.args[2] != -1:
        raise NotImplementedError(f"Only size dimension = 0 is implemented.")

    meta.parameters["common"]["results"]["data_out_0"] = meta.parameters["common"][
        "args"
    ]["data_in_0"]
    meta.parameters["common"]["results"]["data_out_0"]["size"] = [
        math.prod(meta.parameters["common"]["results"]["data_out_0"]["size"]),
    ]

    return meta


# ----------------------------------------------------------
# Size
# ----------------------------------------------------------


def analyse_common_parameters_size(meta):
    """
    Memory transformation.
    """
    meta.parameters["common"]["results"] = {}
    if meta.node.args[1] != 0:
        raise NotImplementedError(f"Only size dimension = 0 is implemented.")

    meta.parameters["common"]["results"]["data_out_0"] = meta.parameters["common"][
        "args"
    ]["data_in_0"]
    meta.parameters["common"]["results"]["data_out_0"]["size"] = [
        math.prod(meta.parameters["common"]["results"]["data_out_0"]["size"]),
    ]

    return meta


# ----------------------------------------------------------
# Flatten
# ----------------------------------------------------------


def analyse_common_parameters_flatten(meta):
    """
    Memory transformation.
    """
    meta.parameters["common"]["results"] = {}
    start_dim = meta.node.kwargs["start_dim"]
    end_dim = meta.node.kwargs["end_dim"]
    # TODO: Implement a flattening function for the shape
    if start_dim != 1 or end_dim != -1:
        raise NotImplementedError(f"Complex flatten function is not implemented yet...")
    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": [
                math.prod(meta.parameters["common"]["args"]["data_in_0"]["size"]),
                1,
            ],
        }
    }

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
    if meta.module is not None:
        for name, parameter in meta.module.named_parameters():
            meta.parameters["common"]["args"][name] = {
                "type": "float",
                "precision": [32],
                "size": parameter.shape,
                "from": None,
            }
    else:
        meta.parameters["common"]["args"]["weight"] = meta.parameters["common"][
            "args"
        ].pop("data_in_1")

    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": (
                meta.parameters["common"]["args"]["weight"]["size"][0],
                meta.parameters["common"]["args"]["data_in_0"]["size"][1],
            ),
        }
    }
    return meta


# ----------------------------------------------------------
# ReLU
# ----------------------------------------------------------


def analyse_common_parameters_relu(meta):
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": meta.parameters["common"]["args"]["data_in_0"]["size"],
        }
    }
    return meta


# ----------------------------------------------------------
# Constant
# ----------------------------------------------------------


def _fetch_attr(target: str, meta):
    """
    Get attr return tensor
    """

    target_atoms = target.split(".")
    attr_itr = meta.model
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def analyse_common_parameters_constant(meta):
    """
    A constant is an op that provides constant.
    """

    result = _fetch_attr(meta.node.target, meta)

    meta.parameters["common"]["args"] = {}
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"]["data_out_0"] = {
        "type": "float",
        "precision": [32],
        "size": list(result.size()),
    }
    return meta
