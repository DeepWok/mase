import inspect
import math

import torch
import inspect
from chop.tools.utils import to_numpy_if_tensor as to_numpy
from chop.passes.graph.utils import vf, get_node_by_name
import traceback


# ----------------------------------------------------------
# Utility
# ----------------------------------------------------------

# The following information is fetched from pytorch documentation
func_data = {
    # https://pytorch.org/docs/stable/generated/torch.flatten.html#torch.flatten
    "flatten": {"input": "data_in", "start_dim": "config", "end_dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
    "relu": {"input": "data_in", "inplace": "config"},
    # https://pytorch.org/docs/stable/generated/torch.add.html
    "add": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.mul.html
    "mul": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.sub.html
    "sub": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.matmul.html
    "matmul": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.bmm.html
    "bmm": {"input": "data_in", "mat2": "data_in"},
}

module_data = {
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html#MaxPool1d
    "adaptive_avg_pool1d": {"input": "data_in"},
    "adaptive_avg_pool2d": {"input": "data_in"},
    "adaptive_max_pool1d": {"input": "data_in"},
    "adaptive_max_pool2d": {"input": "data_in"},
    "avg_pool1d": {"input": "data_in"},
    "avg_pool2d": {"input": "data_in"},
    "max_pool1d": {"input": "data_in"},
    "max_pool2d": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d
    "batch_norm1d": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm1d
    "batch_norm2d": {"input": "data_in"},
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d
    "conv1d": {"input": "data_in"},
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
    "conv2d": {"input": "data_in"},
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d
    "conv3d": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    "layer_norm": {"input": "data_in"},
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    "linear": {"input": "data_in"},
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#ReLU
    "relu": {"input": "data_in"},
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Hardtanh
    "hardtanh": {"input": "data_in"},
    "relu6": {"input": "data_in"},
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout
    "dropout": {"input": "data_in"},
    "hardswish": {"input": "data_in"},
    "hardsigmoid": {"input": "data_in"},
    # TODO: check this
    "attention": {"input": "data_in"},
}


method_data = {
    # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
    # view can have arbitary shape, goes from shape_0 ... shape_n, we support up to 4-dim
    "view": {
        "shape_0": "data_in",
        "shape_1": "data_in",
        "shape_2": "data_in",
        "shape_3": "data_in",
    },
    # https://pytorch.org/docs/stable/generated/torch.Tensor.addmm.html#torch.Tensor.addmm
    "addm": {"mat1": "data_in", "mat2": "data_in", "beta": "config", "alpha": "config"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.size.html#torch.Tensor.size
    "size": {"dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.shape.html#torch.Tensor.shape
    "shape": {"dim": "config"},
}


def match_args_and_kwargs(meta, args, kwargs, data, add_value):
    ordered_func_data = [(k, v) for k, v in data.items()]
    meta.parameters["common"]["args"] = {}
    meta_kwargs = {}
    j = 0

    for i, x in enumerate(args):
        if isinstance(x, torch.Tensor) and ordered_func_data[i][1] == "data_in":
            arg_meta = {
                "shape": list(x.shape),
                "torch_dtype": x.dtype,
                "type": "float",
                "precision": [32],
            }
            if add_value:
                arg_meta["value"] = x
            meta.parameters["common"]["args"][f"data_in_{j}"] = arg_meta
            j += 1
        else:
            # this is not an data_in, but just actually an named arg
            n, vtype = ordered_func_data[i]
            meta_kwargs[n] = args[i]

    for k, v in kwargs.items():
        if data[k] == "data_in":
            # rename this to mase data_in_number
            arg_meta = {
                "shape": list(v.shape),
                "torch_dtype": v.dtype,
                "type": "float",
                "precision": [32],
            }
            if add_value:
                arg_meta["value"] = v
            meta.parameters["common"]["args"][f"data_in_{j}"] = arg_meta
            j += 1
        else:
            # otherwise this must be a configuration parameter in meta
            meta_kwargs[k] = v
    # merge configuratipn args
    meta.parameters["common"]["args"] = meta.parameters["common"]["args"] | meta_kwargs
    return meta


def analyse_result(meta, result, add_value):
    # deal with results
    meta.parameters["common"]["results"] = {}
    if isinstance(result, torch.Tensor):
        meta.parameters["common"]["results"]["data_out_0"] = {
            "type": "float",
            "precision": [32],
            "shape": list(result.shape),
            "torch_dtype": result.dtype,
        }
        if add_value:
            meta.parameters["common"]["results"]["data_out_0"]["value"] = result
    else:
        meta.parameters["common"]["results"]["data_out_0"] = {
            "type": type(result),
            "shape": [1],
            "value": result,
        }
    return meta


# ----------------------------------------------------------
# Placeholder
# ----------------------------------------------------------


def analyse_common_parameters_placeholder(meta, result, args, kwargs, add_value=True):
    """
    The placeholder itself does not contain any information, but can be provided from users.
    """
    var_name = meta.node.target
    # deal with model specific inputs, normally these are not numerical values/tensors
    if var_name in meta.model.additional_inputs:
        meta.parameters["common"]["args"] = {}
        meta.parameters["common"]["results"] = {}
        meta.parameters["common"]["results"]["data_out_0"] = {
            "type": "model_specific_input",
            "shape": result.shape,
            "torhc_dtype": result.dtype,
        }
        if add_value:
            meta.parameters["common"]["results"]["data_out_0"]["value"] = result
        return meta

    meta.parameters["common"]["args"] = {}
    meta = analyse_result(meta, result, add_value)
    return meta


# ----------------------------------------------------------
# Function
# ----------------------------------------------------------


def analyse_common_parameters_function(meta, result, args, kwargs, add_value=True):
    # fetch mase info
    mase_op = meta.parameters["common"]["mase_op"]

    # deal with result
    meta = analyse_result(meta, result, add_value)
    # deal with args and kwargs
    meta = match_args_and_kwargs(meta, args, kwargs, func_data[mase_op], add_value)

    return meta


# ----------------------------------------------------------
# Module
# ----------------------------------------------------------


def analyse_common_parameters_module(meta, result, args, kwargs, add_value=True):
    mase_op = meta.parameters["common"]["mase_op"]
    meta = match_args_and_kwargs(meta, args, kwargs, module_data[mase_op], add_value)
    for name, parameter in meta.module.named_parameters():
        meta.parameters["common"]["args"][name] = {
            "type": "float",
            "precision": [32],
            "shape": list(parameter.shape),
            "from": None,
        }
        if add_value:
            meta.parameters["common"]["args"][name]["value"] = parameter

    meta = analyse_result(meta, result, add_value)
    return meta


# ----------------------------------------------------------
# Module
# ----------------------------------------------------------


def analyse_common_parameters_method(meta, result, args, kwargs, add_value=True):
    mase_op = meta.parameters["common"]["mase_op"]
    meta = analyse_result(meta, result, add_value)
    meta = match_args_and_kwargs(meta, args, kwargs, method_data[mase_op], add_value)
    return meta


# ----------------------------------------------------------
# Attribute
# ----------------------------------------------------------


def analyse_common_parameters_attr(meta, result, args, kwargs, add_value=True):
    meta.parameters["common"]["args"] = {}
    meta = analyse_result(meta, result, add_value)
    return meta


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------


def analyse_common_parameters_output(meta, result, args, kwargs, add_value=True):
    meta.parameters["common"]["args"] = {}
    meta = analyse_result(meta, result, add_value)
    return meta
