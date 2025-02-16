import inspect
from collections import OrderedDict
from functools import reduce

import torch

from chop.nn.quantized.modules import quantized_module_map

from chop.ir.graph import MaseMetadata
from chop.tools import get_logger

logger = get_logger(__name__)
logger.setLevel("INFO")

# ----------------------------------------------------------
# Utility
# ----------------------------------------------------------

# The following information is fetched from pytorch documentation
func_data = {
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    "scaled_dot_product_attention": {
        "query": "data_in",
        "key": "data_in",
        "value": "data_in",
        "attn_mask": "data_in",
        "dropout_p": "config",
        "is_causal": "config",
        "scale": "config",
    },
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
    # https://pytorch.org/docs/stable/generated/torch.addmm.html
    "baddbmm": {
        "input": "data_in",
        "batch1": "data_in",
        "batch2": "data_in",
        "beta": "config",
        "alpha": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.addmm.html
    "addmm": {
        "input": "data_in",
        "mat1": "data_in",
        "mat2": "data_in",
        "beta": "config",
        "alpha": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.add.html
    "add": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.mul.html
    "mul": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.sub.html
    "sub": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.matmul.html
    "matmul": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.mm.html
    "mm": {"input": "data_in", "mat2": "data_in"},
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
    "floor": {"input": "data_in"},
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
    # https://pytorch.org/docs/stable/generated/torch.greater.html
    "greater": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.gt.html
    "gt": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.abs.html
    "abs": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.sigmoid.html
    "sigmoid": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.argmax.html
    "argmax": {"input": "data_in"},
    # dataflow_split
    "df_split": {"x": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.split.html
    "split": {"input": "data_in", "split_size_or_sections": "config", "dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.logical_not.html
    "not": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.tile.html
    "tile": {"input": "data_in", "dims": "config"},
    # https://pytorch.org/docs/stable/generated/torch.lt.html#torch.lt
    "less": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.lt.html#torch.lt
    "lt": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.le.html
    "lessorequal": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.le.html
    "le": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.min.html
    "min": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.neg.html
    "neg": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.log.html
    "log": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.mean.html
    "mean": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.arange.html
    "arange": {
        "start": "config",
        "end": "config",
        "step": "config",
        "dtype": "config",
        "device": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.range.html
    "range": {"start": "config", "end": "config", "step": "config"},
    # https://pytorch.org/docs/stable/generated/torch.where.html
    "where": {"condition": "config", "input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.equal.html
    "eq": {"input": "data_in", "other": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.ne.html
    "ne": {"input": "data_in", "other": "data_in"},
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
    "full": {"size": "config", "fill_value": "data_in", "device": "config"},
    # get item
    "getitem": {"in": "data_in", "select": "config"},
    # getattr
    "getattr": {"a": "data_in", "b": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.ones.html
    "ones": {"size": "config", "device": "config"},
    "finfo": {"dtype": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    "layer_norm": {
        "input": "data_in",
        "normalized_shape": "config",
        "weight": "data_in",
        "bias": "data_in",
        "eps": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.transpose.html
    "transpose": {"input": "data_in", "dim_0": "config", "dim_1": "config"},
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
    "embedding": {
        "input": "data_in",
        "weight": "data_in",
        "padding_idx": "config",
        "max_norm": "config",
        "norm_type": "config",
        "scale_grad_by_freq": "config",
        "sparse": "config",
    },
    # Inserted ops from the replace_method_with_function pass
    "torch_size": {"input": "data_in", "dim": "config"},
    "torch_contiguous": {
        "input": "data_in",
        "memory_format": "config",
    },
    # arbitrary length - support up to 4
    "torch_expand": {
        "input": "data_in",
        "size_0": "config",
        "size_1": "config",
        "size_2": "config",
        "size_3": "config",
    },
    "torch_view": {
        "input": "data_in",
        "shape_0": "config",
        "shape_1": "config",
        "shape_2": "config",
        "shape_3": "config",
    },
    "torch_reshape": {
        "input": "data_in",
        "shape_0": "config",
        "shape_1": "config",
        "shape_2": "config",
        "shape_3": "config",
    },
    "torch_split": {
        "input": "data_in",
        "split_size": "config",
        "dim": "config",
    },
    "torch_permute": {
        "input": "data_in",
        "dim_0": "config",
        "dim_1": "config",
        "dim_2": "config",
        "dim_3": "config",
    },
    "torch_transpose": {
        "input": "data_in",
        "dim0": "config",
        "dim1": "config",
    },
    # DTensor ops
    "dtensor_arange": {
        "device_mesh": "config",
        "start": "config",
        "end": "config",
        "step": "config",
        "out": "config",
        "dtype": "config",
        "layout": "config",
        "device": "config",
        "requires_grad": "config",
    },
    # tensor constructor
    "tensor": {
        "data": "data_in",
    },
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html
    "dropout": {
        "input": "data_in",
        "p": "config",
        "training": "config",
        "inplace": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_avg_pool1d.html
    "adaptive_avg_pool1d": {"input": "data_in", "output_size": "config"},
    "adaptive_avg_pool2d": {"input": "data_in", "output_size": "config"},
    "adaptive_max_pool1d": {"input": "data_in", "output_size": "config"},
    "adaptive_max_pool2d": {"input": "data_in", "output_size": "config"},
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
    "gelu": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    "crossentropyloss": {
        "input": "data_in",
        "target": "data_in",
    },
    # chop.nn.modules.lora.LoRALinear
    "loralinear": {"input": "data_in"},
    "grouped_query_attention": {"input": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#flatten
    "flatten": {"input": "data_in", "start_dim": "config", "end_dim": "config"},
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
    # https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html#torch.Tensor.reshape
    "reshape": {
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
        "size_4": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
    "reshape": {
        "size_0": "config",
        "size_1": "config",
        "size_2": "config",
        "size_3": "config",
    },
    # Tensor.max(dim=None, keepdim=False)
    "max": {
        "dim": "config",
        "keepdim": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.Tensor.sum.html
    "sum": {
        "dim": "config",
        "keepdim": "config",
    },
    # https://pytorch.org/docs/stable/generated/torch.Tensor.round.html
    "round": {},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.floor.html
    "floor": {},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.clamp.html
    "clamp": {
        "min": "config",
        "max": "config",
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
    # https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html#torch.Tensor.masked_fill
    "masked_fill": {"mask": "data_in", "value": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
    "masked_fill_": {"mask": "data_in", "value": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html#torch.Tensor.unsqueeze
    "unsqueeze": {"input": "data_in", "dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.split.html#torch.Tensor.split
    "split": {"input": "data_in", "split_size_or_sections": "config", "dim": "config"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.bool.html
    "bool": {"memory_format": "config"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.long.html
    "long": {"memory_format": "config"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.type_as.html
    "type_as": {"tensor": "data_in"},
    # https://pytorch.org/docs/stable/generated/torch.Tensor.index_select.html
    "index_select": {
        "input": "data_in",
        "dim": "config",
        "index": "data_in",
    },
    # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
    "detach": {"input": "data_in"},
}

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------


def get_type_and_precision(meta):
    # * Fetch type and precision from q_config for quantized modules
    if isinstance(meta.module, tuple(quantized_module_map.values())):
        cf = (
            meta.module.q_config
            if hasattr(meta.module, "q_config")
            else meta.module.config
        )
        arg_type = "fixed"
        arg_precision = [
            cf["data_in_width"],
            cf["data_in_frac_width"],
        ]
    else:
        arg_type = "float"
        arg_precision = [32]
    return arg_type, arg_precision


def get_shape(x):
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return list(x.shape)
    elif isinstance(x, int):
        return [1]
    elif isinstance(x, (list, tuple, torch.Size)):
        return [len(x)]
    else:
        return [0]


def deepgetattr(obj, attr):
    """Recurses through an attribute chain to get the ultimate value."""
    return reduce(getattr, attr.split("."), obj)


# ----------------------------------------------------------
# Metadata annotators
# ----------------------------------------------------------


def _annotate_arg_metadata(
    meta: MaseMetadata,
    args: list,
    kwargs: dict,
    func_data: dict,
    add_value: bool,
):
    """
    Analyse target args and kwargs received from shape propagation to annotate combined meta["mase"]["args"]
    dictionary with metadata about each argument. The order of the args and kwargs must be preserved in the
    combined dictionary (this is expected by downstream passes). However, arguments with the 'data_in' flag
    in func_data are renamed to 'data_in_{itr}' where itr = 0 ... the number of data_in arguments.

    This function should not be called directly, but rather through the `annotate_common_parameters_<OP>` function.
    The value in the meta["common"]["args"] dictionary should always be a dictionary, not a tensor.

    Args:
        meta (MaseMetadata): The metadata object.
        args (list): List of args passed to the target.
        kwargs (dict): Dictionary of kwargs passed to the target.
        func_data (dict): Dictionary defining whether each argument is data_in or config.
        add_value (bool): indicate whether to add the value of the tensor to the metadata.

    Returns:
        MaseMetadata: metadata object with annotated args.
    """
    ordered_func_data = [(k, v) for k, v in func_data.items()]
    meta["common"]["args"] = OrderedDict()
    data_in_itr = 0

    arg_type, arg_precision = get_type_and_precision(meta)

    # * Handle args
    for i, x in enumerate(args):

        # Input data tensor
        if isinstance(x, torch.Tensor) and ordered_func_data[i][1] == "data_in":
            arg_meta = {
                "shape": list(x.shape),
                "torch_dtype": x.dtype,
                "type": arg_type,
                "precision": arg_precision,
            }
            if add_value:
                arg_meta["value"] = x
            meta["common"]["args"][f"data_in_{data_in_itr}"] = arg_meta
            data_in_itr += 1

        # Tuple of tensors
        elif isinstance(x, tuple) and all([isinstance(x, torch.Tensor) for x in x]):
            for k, x in enumerate(x):
                arg_meta = {
                    "shape": list(x.shape),
                    "torch_dtype": x.dtype,
                    "type": arg_type,
                    "precision": arg_precision,
                }
                if add_value:
                    arg_meta["value"] = x
                meta["common"]["args"][f"data_in_{data_in_itr}"] = arg_meta
                data_in_itr += 1
        # Unknown data_in type or config argument
        else:
            # Don't increment the iterator for config arguments, but
            # preserve order in meta["common"]["args"]
            arg_name, arg_flag = ordered_func_data[i]

            if arg_flag == "data_in":
                arg_name = f"data_in_{data_in_itr}"
                data_in_itr += 1

            meta["common"]["args"][arg_name] = {
                "torch_dtype": x.dtype if isinstance(x, torch.Tensor) else None,
                "type": type(args[i]),
                "precision": arg_precision,
                "shape": get_shape(args[i]),
            }

            if add_value:
                meta["common"]["args"][arg_name]["value"] = args[i]

    # * Handle kwargs
    for k, v in kwargs.items():
        if func_data[k] == "data_in":
            # rename this to mase data_in_number
            shape = get_shape(v)
            arg_meta = {
                "shape": shape,
                "torch_dtype": v.dtype if isinstance(v, torch.Tensor) else type(v),
                "type": arg_type,
                "precision": arg_precision,
            }
            if add_value:
                arg_meta["value"] = v
            meta["common"]["args"][f"data_in_{data_in_itr}"] = arg_meta
            data_in_itr += 1
        elif k == "inplace":
            # although inplace is marked as a config type, it is in fact just a boolean flag
            meta["common"]["args"][k] = v
        else:
            # otherwise this must be a configuration parameter in meta
            # meta_kwargs[k] = v
            meta["common"]["args"][k] = {
                "type": type(v),
                "precision": arg_precision,
                "shape": get_shape(v),
            }
            if add_value:
                meta["common"]["args"][k]["value"] = v

    return meta


def _annotate_result_metadata(
    meta: MaseMetadata,
    result,
    add_value: bool,
) -> MaseMetadata:
    """
    Analyse the result from running the target to annotate the meta["mase"]["results"] dictionary with metadata.

    Args:
        meta (MaseMetadata): The metadata object.
        result (_type_): The result object.
        add_value (bool): indicate whether to add the value of the tensor to the metadata.

    Returns:
        MaseMetadata: metadata object with annotated results.
    """
    # deal with results
    meta["common"]["results"] = OrderedDict()

    result_type, result_precision = get_type_and_precision(meta)

    if isinstance(result, torch.Tensor):
        meta["common"]["results"]["data_out_0"] = {
            "type": result_type,
            "precision": result_precision,
            "shape": list(result.shape),
            "torch_dtype": result.dtype,
        }
        if add_value:
            meta["common"]["results"]["data_out_0"]["value"] = result

    # check if it's a tuple of tensors
    elif isinstance(result, tuple) and all(
        [isinstance(x, torch.Tensor) for x in result]
    ):
        for i, x in enumerate(result):
            meta["common"]["results"][f"data_out_{i}"] = {
                "type": result_type,
                "precision": result_precision,
                "shape": list(x.shape),
                "torch_dtype": x.dtype,
            }
            if add_value:
                meta["common"]["results"][f"data_out_{i}"]["value"] = x
    else:
        logger.debug(
            f"Expected result to be a tensor or tuple of tensors, but found: {type(result)}. Will annotate with default value, but this may cause issues downstream."
        )
        meta["common"]["results"]["data_out_0"] = {
            "type": type(result),
            "shape": [1],
        }
        if add_value:
            meta["common"]["results"]["data_out_0"]["value"] = result

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
        meta["common"]["args"] = {}
        meta["common"]["results"] = {}
        meta["common"]["results"]["data_out_0"] = {
            "type": "model_specific_input",
            "shape": result.shape,
            "torhc_dtype": result.dtype,
        }
        if add_value:
            meta["common"]["results"]["data_out_0"]["value"] = result
        return meta

    meta["common"]["args"] = {}
    meta = _annotate_result_metadata(meta, result, add_value)
    return meta


# ----------------------------------------------------------
# Function
# ----------------------------------------------------------


def analyse_common_parameters_function(meta, result, args, kwargs, add_value=True):
    # fetch mase info
    mase_op = meta["common"]["mase_op"]

    # deal with result
    meta = _annotate_result_metadata(meta, result, add_value)
    # deal with args and kwargs
    meta = _annotate_arg_metadata(meta, args, kwargs, func_data[mase_op], add_value)

    return meta


# ----------------------------------------------------------
# Module
# ----------------------------------------------------------


def analyse_common_parameters_module(meta, result, args, kwargs, add_value=True):
    mase_op = meta["common"]["mase_op"]
    node_module = deepgetattr(meta.model, meta.node.target)

    if mase_op == "user_defined_module":
        for custom_module, v in meta.model.custom_ops["modules"].items():
            if isinstance(node_module, custom_module):
                module_args = v["args"]
                break
    else:
        module_args = module_data[mase_op]

    meta = _annotate_arg_metadata(meta, args, kwargs, module_args, add_value)

    arg_type, arg_precision = get_type_and_precision(meta)

    for name, parameter in meta.module.named_parameters():
        name = name.replace(".", "_")
        meta["common"]["args"][name] = {
            "type": arg_type,
            "precision": arg_precision,
            "shape": (
                list(parameter.shape)
                if len(parameter.shape) > 1
                else list(parameter.unsqueeze(dim=0).shape)
            ),
            "from": None,
        }
        if add_value:
            meta["common"]["args"][name]["value"] = parameter

    meta = _annotate_result_metadata(meta, result, add_value)
    return meta


# ----------------------------------------------------------
# Method
# ----------------------------------------------------------


def analyse_common_parameters_method(meta, result, args, kwargs, add_value=True):
    mase_op = meta["common"]["mase_op"]
    meta = _annotate_result_metadata(meta, result, add_value)
    meta = _annotate_arg_metadata(meta, args, kwargs, method_data[mase_op], add_value)
    return meta


# ----------------------------------------------------------
# Attribute
# ----------------------------------------------------------


def analyse_common_parameters_attr(meta, result, args, kwargs, add_value=True):
    meta["common"]["args"] = {}
    meta = _annotate_result_metadata(meta, result, add_value)
    return meta


# ----------------------------------------------------------
# Output
# ----------------------------------------------------------


def analyse_common_parameters_output(meta, result, args, kwargs, add_value=True):
    meta["common"]["args"] = {}
    meta = _annotate_result_metadata(meta, result, add_value)
    return meta
