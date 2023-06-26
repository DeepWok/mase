"""
Some of the functions here are taken from the Modifier class we had before
"""
import torch
from torch import nn
from typing import Dict

from chop.tools.utils import copy_weights
from .quantized_funcs import quantized_func_map
from .quantized_modules import quantized_module_map

type_to_name_map = {
    nn.Linear: "linear",
    nn.Conv1d: "conv1d",
    nn.Conv2d: "conv2d",
}

def create_new_module(original_module: nn.Module, config: Dict):
    original_module_cls = type(original_module)
    quan_name = config.get("name")

    if original_module_cls is nn.Linear:
        new_module_cls = quantized_module_map[f'linear_{quan_name}']
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
    elif original_module_cls in (nn.Conv1d, nn.Conv2d):
        name = f'conv1d_{quan_name}' if original_module_cls is nn.Conv1d else f'conv2d_{quan_name}'
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

    elif original_module_cls is nn.ReLU:
        new_module_cls = quantized_module_map[f'relu_{quan_name}']
        new_module = new_module_cls(
            inplace=original_module.inplace, config=config)

    elif original_module_cls is nn.AvgPool2d:
        new_module_cls = quantized_module_map[f'avgpool2d_{quan_name}']
        new_module = new_module_cls(
            kernel_size=original_module.kernel_size,
            stride=original_module.stride,
            padding=original_module.padding,
            ceil_mode=original_module.ceil_mode,
            count_include_pad=original_module.count_include_pad,
            divisor_override=original_module.divisor_override,
            config=config,
        )
    elif original_module_cls is nn.AdaptiveAvgPool2d:
        new_module_cls = quantized_module_map[f'adaptiveavgpool2d_{quan_name}']
        new_module = new_module_cls(
            output_size=original_module.output_size, config=config
        )
    else:
        raise NotImplementedError(
            f"Unsupported module class {original_module_cls} to modify"
        )
    return new_module


def create_new_fn(node, config: Dict):
    mase_op = node.mase_op
    quan_name = config.get("name")
    func_name = f'{mase_op}_{quan_name}'
    new_func = quantized_func_map[func_name]
    args, kwargs = node.args, node.kwargs | config
    return new_func, args, kwargs