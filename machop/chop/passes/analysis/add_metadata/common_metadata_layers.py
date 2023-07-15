import inspect
import math

import torch
from chop.passes.utils import vf, get_node_by_name

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
# Size
# ----------------------------------------------------------


def analyse_common_parameters_size(meta):
    """
    Size is an attribute which provides a constrant value in a static graph.
    """
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"]["data_out_0"] = {
        "type": "float",
        "precision": [32],
        "size": [1],
        "value": meta.parameters["common"]["args"]["data_in_0"]["size"][
            meta.node.args[1]
        ],
    }
    return meta


# ----------------------------------------------------------
# t
# ----------------------------------------------------------


def analyse_common_parameters_t(meta):
    """
    Memory transformation.
    """
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": meta.parameters["common"]["args"]["data_in_0"]["size"],
        }
    }
    (
        meta.parameters["common"]["results"]["data_out_0"]["size"][0],
        meta.parameters["common"]["results"]["data_out_0"]["size"][1],
    ) = (
        meta.parameters["common"]["results"]["data_out_0"]["size"][1],
        meta.parameters["common"]["results"]["data_out_0"]["size"][0],
    )

    return meta


# ----------------------------------------------------------
# t
# ----------------------------------------------------------


def analyse_common_parameters_t(meta):
    """
    Memory transformation.
    """
    meta.parameters["common"]["results"] = {}
    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": meta.parameters["common"]["args"]["data_in_0"]["size"],
        }
    }
    (
        meta.parameters["common"]["results"]["data_out_0"]["size"][0],
        meta.parameters["common"]["results"]["data_out_0"]["size"][1],
    ) = (
        meta.parameters["common"]["results"]["data_out_0"]["size"][1],
        meta.parameters["common"]["results"]["data_out_0"]["size"][0],
    )

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
                "size": list(parameter.shape),
                "from": None,
            }
    else:
        meta.parameters["common"]["args"]["weight"] = meta.parameters["common"][
            "args"
        ].pop("data_in_1")
        meta.parameters["common"]["args"]["bias"] = meta.parameters["common"][
            "args"
        ].pop("data_in_2")

    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": [
                meta.parameters["common"]["args"]["data_in_0"]["size"][0],
                meta.parameters["common"]["args"]["weight"]["size"][0],
            ],
        }
    }
    return meta


# ----------------------------------------------------------
# ReLU, Binary Op
# ----------------------------------------------------------


def analyse_common_parameters_pass(meta):
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


# ----------------------------------------------------------
# View
# ----------------------------------------------------------


def _load_arg(meta):
    return torch.fx.graph.map_arg(
        meta.node.args, lambda n: get_node_by_name(n.meta["mase"].graph, n.name)
    )


def _load_kwarg(meta):
    return torch.fx.graph.map_arg(
        meta.node.kwargs, lambda n: get_node_by_name(n.meta["mase"].graph, n.name)
    )


def analyse_common_parameters_view(meta):
    """
    Memory transformation.
    The current approach is just to run inference and directly fetch the result size.
    TODO: This needs to be replaced with direct shape inference
    """

    self_obj, *args = _load_arg(meta)
    dummy_data = torch.full(
        self_obj.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"], 1
    )

    shape = []
    for arg in args:
        if isinstance(arg, torch.fx.Node):
            assert (
                "value"
                in arg.meta["mase"].parameters["common"]["results"]["data_out_0"].keys()
            ), "Try to interpret a non-constant node. This is not expected"
            shape.append(
                arg.meta["mase"].parameters["common"]["results"]["data_out_0"]["value"]
            )
        else:
            shape.append(arg)
    size = list(dummy_data.view(shape).size())

    meta.parameters["common"]["results"] = {}
    arg = meta.parameters["common"]["args"]["data_in_0"]
    meta.parameters["common"]["results"]["data_out_0"] = {
        "type": arg["type"],
        "precision": arg["precision"],
        "size": size,
    }

    return meta


# ----------------------------------------------------------
# General Modules
# ----------------------------------------------------------


def _get_size_by_module_simulation(meta):
    """
    Otain the size of the output by executing the module
    """
    self_obj, *args = _load_arg(meta)
    kwargs = _load_kwarg(meta)

    dummy_data = torch.full(
        self_obj.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"], 1.0
    )
    result = meta.module(dummy_data, *args, **kwargs)
    size = list(result.size())
    return size


def analyse_common_parameters_module(meta):
    for name, parameter in meta.module.named_parameters():
        meta.parameters["common"]["args"][name] = {
            "type": "float",
            "precision": [32],
            "size": list(parameter.shape),
            "from": None,
        }
    size = _get_size_by_module_simulation(meta)

    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": size,
        }
    }
    return meta


# ----------------------------------------------------------
# General Functions
# ----------------------------------------------------------


def _get_size_by_function_simulation(meta):
    """
    Otain the size of the output by executing the function
    """
    self_obj, *args = _load_arg(meta)
    kwargs = _load_kwarg(meta)

    dummy_data = torch.full(
        self_obj.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"], 1.0
    )
    args_val = [
        torch.full(
            a.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"], 1.0
        )
        if isinstance(a, torch.fx.Node)
        else a
        for a in args
    ]
    kwargs_val = {}
    for key, val in kwargs.items():
        if isinstance(val, torch.fx.Node):
            kwargs_val[key] = torch.full(
                val.meta["mase"].parameters["common"]["results"]["data_out_0"]["size"],
                1.0,
            )
        else:
            kwargs_val[key] = val

    result = meta.node.target(dummy_data, *args_val, **kwargs_val)
    size = list(result.size())
    return size


def analyse_common_parameters_function(meta):
    size = _get_size_by_function_simulation(meta)
    meta.parameters["common"]["results"] = {
        "data_out_0": {
            "type": "float",
            "precision": [32],
            "size": size,
        }
    }
    return meta


# ----------------------------------------------------------
# Conv1d
# ----------------------------------------------------------


def analyse_common_parameters_conv1d(meta):
    if meta.module is not None:
        return analyse_common_parameters_module(meta)
    else:
        assert (
            False
        ), "torch.functional.conv1d not supported. Please use nn.Module.Conv1d for now"
    return meta


# ----------------------------------------------------------
# Conv2d
# ----------------------------------------------------------


def analyse_common_parameters_conv2d(meta):
    if meta.module is not None:
        return analyse_common_parameters_module(meta)
    else:
        assert (
            False
        ), "torch.functional.conv2d not supported. Please use nn.Module.Conv2d for now"

    return meta
