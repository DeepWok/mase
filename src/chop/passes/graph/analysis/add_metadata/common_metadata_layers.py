import inspect
import math

import torch
import inspect
from chop.tools.utils import to_numpy_if_tensor as to_numpy
from chop.passes.graph.utils import vf, get_node_by_name
import traceback
from functools import reduce


# ----------------------------------------------------------
# Utility
# ----------------------------------------------------------

# The following information is fetched from pytorch documentation
func_data = {
    # https://pytorch.org/docs/stable/generated/torch.flatten.html#torch.flatten
    "flatten": {"input": "data_in", "start_dim": "config", "end_dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
    "relu": {"input": "data_in", "inplace": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.hardshrink.html
    "hardshrink": {"input": "data_in", "lambd": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html
    "silu": {"input": "data_in", "inplace": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.elu.html
    "elu": {"input": "data_in", "alpha": "config", "inplace": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html
    "sigmoid": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.softshrink.html
    "softshrink": {"input": "data_in", "lambd": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html
    "logsigmoid": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
    "softmax": {"input": "data_in", "dim": "config", "dtype": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
    "selu": {"input": "data_in", "inplace": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
    "tanh": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    "gelu": {"input": "data_in", "inplace": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
    "softsign": {"input": "data_in", "inplace": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    "softplus": {"input": "data_in", "inplace": "config"},
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
    # https://pytorch.org/docs/stable/generated/torch.squeeze.html
    "squeeze": {"input": "data_in", "dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
    "unsqueeze": {"input": "data_in", "dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.gather.html
    "gather": {"input": "data_in", "index": "config", "dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.mean.html
    "mean": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.pow.html
    "pow": {"input": "data_in", "exponent": "config"},
    # https://pytorch.org/docs/stable/generated/torch.sqrt.html
    "sqrt": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.div.html
    "div": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.cat.html
    "cat": {"tensors": "data_in", "dim": "config"},
    # onnx_slice (custom implementation in onnx config)
    "slice": {
        "data": "data_in",
        "starts": "config",
        "ends": "config",
        "axes": "config",
        "steps": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.reshape.html
    "reshape": {"input": "data_in", "shape": "config"},
    # https://pytorch.org/docs/stable/generated/torch.permute.html
    "permute": {"input": "data_in", "dims": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
    "softmax": {
        "input": "data_in",
        "dim": "config",
        "_stacklevel": "config",
        "dtype": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
    "gelu": {"input": "data_in"},
    # https://pytorch.org/docs/stable/special.html#torch.special.erf
    "erf": {"input": "data_in"},
    # onnx_shape (custom implementation)
    "shape": {"input": "data_in"},
    # onnx_identity (custom implementation)
    "identity": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.max.html
    "max": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.sin.html
    "sin": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.cos.html
    "cos": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.tan.html
    "tan": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.tanh.html
    "tanh": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.gt.html#torch.gt
    "greater": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.abs.html
    "abs": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.sigmoid.html
    "sigmoid": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.argmax.html
    "argmax": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.split.html
    "split": {"input": "data_in", "split_size_or_sections": "config", "dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.logical_not.html
    "not": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.tile.html
    "tile": {"input": "data_in", "dims": "config"},
    # https://pytorch.org/docs/stable/generated/torch.lt.html#torch.lt
    "less": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.le.html
    "lessorequal": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.min.html
    "min": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.neg.html
    "neg": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.log.html
    "log": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.mean.html
    "mean": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.range.html
    "range": {"start": "config", "end": "config", "step": "config"},
    # https://pytorch.org/docs/stable/generated/torch.where.html
    "where": {"condition": "config", "input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.equal.html
    "eq": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.cumsum.html
    "cumsum": {"input": "data_in", "dim": "config"},
    # onnx_gemm (custom implementation)
    "gemm": {
        "A": "data_in",
        "B": "data_in",
        "C": "data_in",
        "alpha": "config",
        "beta": "config",
        "transA": "config",
        "transB": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.full.html
    "full": {"size": "config", "fill_value": "data_in"},
    # get item
    "getitem": {"a": "data_in", "b": "data_in"},
    # getattr
    "getattr": {"a": "data_in", "b": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.ones.html
    "ones": {"size": "config", "device": "config"},
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
    # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    "embedding": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    "layer_norm": {"input": "data_in"},
    "group_norm": {"input": "data_in"},
    "instance_norm2d": {"input": "data_in"},
    "rms_norm": {"input": "data_in"},
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
    # https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
    "tanh": {"input": "data_in"},
    "sigmoid": {"input": "data_in"},
    "logsigmoid": {"input": "data_in"},
    "softshrink": {"input": "data_in"},
    "hardshrink": {"input": "data_in"},
    "silu": {"input": "data_in"},
    "elu": {"input": "data_in"},
    "softmax": {"input": "data_in"},
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
    # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    "to": {"dtype": "config"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
    "expand": {
        "size_0": "config",
        "size_1": "config",
        "size_2": "config",
        "size_3": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.Tensor.dim.html
    "dim": {},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.permute.html#torch.Tensor.permute
    "permute": {
        "dim_0": "config",
        "dim_1": "config",
        "dim_2": "config",
        "dim_3": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.Tensor.transpose.html#torch.Tensor.transpose
    "transpose": {"dim_0": "config", "dim_1": "config"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html#torch.Tensor.contiguous
    "contiguous": {},
}


def match_args_and_kwargs(meta, args, kwargs, data, add_value):
    ordered_func_data = [(k, v) for k, v in data.items()]
    meta.parameters["common"]["args"] = {}
    meta_kwargs = {}
    j = 0
    for i, x in enumerate(args):
        pdb_i = i
        pdb_x = x

        if isinstance(x, torch.Tensor) and ordered_func_data[i][1] == "data_in":
            arg_meta = {
                "shape": list(x.shape),
                "torch_dtype": x.dtype,
                "type": "float",
                "precision": [32],
                "source": "fx_arg",
            }
            if add_value:
                arg_meta["value"] = x
            meta.parameters["common"]["args"][f"data_in_{j}"] = arg_meta
            j += 1
        else:
            # this is not an data_in, but just actually an named arg
            n, vtype = ordered_func_data[i]
            meta_kwargs[n] = args[i]

    def get_shape(x):
        if isinstance(x, torch.Tensor):
            return list(x.shape)
        elif isinstance(x, int):
            return [1]
        elif isinstance(x, list):
            return [len(x)]
        else:
            raise ValueError(f"Unknown type {type(x)}")

    for k, v in kwargs.items():
        if data[k] == "data_in":
            # rename this to mase data_in_number
            arg_meta = {
                "shape": get_shape(v),
                "torch_dtype": v.dtype if isinstance(v, torch.Tensor) else type(v),
                "type": "float",
                "precision": [32],
                "source": "fx_arg",
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


def deepgetattr(obj, attr):
    """Recurses through an attribute chain to get the ultimate value."""
    return reduce(getattr, attr.split("."), obj)


def analyse_common_parameters_module(meta, result, args, kwargs, add_value=True):
    mase_op = meta.parameters["common"]["mase_op"]
    node_module = deepgetattr(meta.model, meta.node.target)

    if mase_op == "user_defined_module":
        for custom_module, v in meta.model.custom_ops["modules"].items():
            if isinstance(node_module, custom_module):
                module_args = v["args"]
                break
    else:
        module_args = module_data[mase_op]

    meta = match_args_and_kwargs(meta, args, kwargs, module_args, add_value)

    for name, parameter in meta.module.named_parameters():
        if "layer_norm" in mase_op:
            breakpoint()
        name = name.replace(".", "_")
        meta.parameters["common"]["args"][name] = {
            "type": "float",
            "precision": [32],
            "shape": list(parameter.shape),
            "from": None,
            "source": "module_parameter",
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
