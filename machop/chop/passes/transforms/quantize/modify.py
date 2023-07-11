"""
Some of the functions here are taken from the Modifier class we had before
"""
from typing import Dict

import torch
from chop.tools.utils import copy_weights
from torch import nn

from .quantized_funcs import quantized_func_map
from .quantized_modules import quantized_module_map

type_to_name_map = {
    nn.Linear: "linear",
    nn.Conv1d: "conv1d",
    nn.Conv2d: "conv2d",
}


def create_new_module(mase_op: str, original_module: nn.Module, config: Dict):
    original_module_cls = type(original_module)
    quant_name = config.get("name")

    if mase_op == "linear":
        new_module_cls = quantized_module_map[f"linear_{quant_name}"]
        use_bias = original_module.bias is not None
        new_module = new_module_cls(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=use_bias,
            config=config,
        )
        copy_weights(original_module.weight, new_module.weight)
        if use_bias:
            copy_weights(original_module.bias, new_module.bias)
    elif mase_op in ("conv1d", "conv2d"):
        name = f"{mase_op}_{quant_name}"
        new_module_cls = quantized_module_map[name]
        use_bias = original_module.bias is not None
        new_module = new_module_cls(
            in_channels=original_module.in_channels,
            out_channels=original_module.out_channels,
            kernel_size=original_module.kernel_size,
            stride=original_module.stride,
            padding=original_module.padding,
            dilation=original_module.dilation,
            groups=original_module.groups,
            bias=use_bias,
            padding_mode=original_module.padding_mode,
            config=config,
        )
        copy_weights(original_module.weight, new_module.weight)
        if use_bias:
            copy_weights(original_module.weight, new_module.weight)

    elif mase_op == "relu":
        new_module_cls = quantized_module_map[f"relu_{quant_name}"]
        new_module = new_module_cls(inplace=original_module.inplace, config=config)

    elif mase_op == "avg_pool2d":
        new_module_cls = quantized_module_map[f"avg_pool2d_{quant_name}"]
        new_module = new_module_cls(
            kernel_size=original_module.kernel_size,
            stride=original_module.stride,
            padding=original_module.padding,
            ceil_mode=original_module.ceil_mode,
            count_include_pad=original_module.count_include_pad,
            divisor_override=original_module.divisor_override,
            config=config,
        )
    elif mase_op == "adaptive_avg_pool2d":
        new_module_cls = quantized_module_map[f"adaptive_avg_pool2d_{quant_name}"]
        new_module = new_module_cls(
            output_size=original_module.output_size, config=config
        )
    else:
        raise NotImplementedError(
            f"Unsupported module class {original_module_cls} to modify"
        )
    return new_module


def create_new_fn(node, config: Dict):
    mase_op = node.meta["mase"].parameters["common"]["mase_op"]
    quant_name = config.get("name")
    func_name = f"{mase_op}_{quant_name}"
    new_func = quantized_func_map[func_name]
    args, kwargs = node.args, node.kwargs | {"config": config}
    return new_func, args, kwargs
