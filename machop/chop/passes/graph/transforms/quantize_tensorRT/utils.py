"""
Some of the functions here are taken from the Modifier class we had before
"""

from typing import Dict

import torch
from chop.tools.utils import copy_weights, init_LinearLUT_weight, init_Conv2dLUT_weight
from torch import nn
import numpy as np

import pytorch_quantization.calib as calib
import pytorch_quantization.nn as qnn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor


def create_new_module(
    mase_op: str,
    original_module: nn.Module,
    config: dict,
):
    original_module_cls = type(original_module)

    if mase_op == "linear":
        use_bias = original_module.bias is not None

        if config.get("name") == "int":
            if "input" in config:
                quant_desc_input = QuantDescriptor(
                    calib_method=config["input"]["calibrator"],
                    axis=config["input"]["quantize_axis"],
                    fake_quant=config["FakeQuantize"],
                )
                qnn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
            if "weight" in config:
                quant_desc_weight = QuantDescriptor(
                    calib_method=config["weight"]["calibrator"],
                    axis=config["weight"]["quantize_axis"],
                    fake_quant=config["FakeQuantize"],
                )
                qnn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

            new_module = qnn.QuantLinear(
                in_features=original_module.in_features,
                out_features=original_module.out_features,
                bias=use_bias,
            )

            copy_weights(original_module.weight, new_module.weight)
            if use_bias:
                copy_weights(original_module.bias, new_module.bias)

        elif config.get("name") == "fp16":
            new_module = original_module.half()

    elif mase_op in ("conv2d"):
        if config.get("name") == "int":
            use_bias = original_module.bias is not None

            if "input" in config:
                quant_desc_input = QuantDescriptor(
                    calib_method=config["input"]["calibrator"],
                    axis=config["input"]["quantize_axis"],
                    fake_quant=config["FakeQuantize"],
                )
                qnn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
            if "weight" in config:
                quant_desc_weight = QuantDescriptor(
                    calib_method=config["weight"]["calibrator"],
                    axis=config["weight"]["quantize_axis"],
                    fake_quant=config["FakeQuantize"],
                )
                qnn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)

            new_module = qnn.QuantConv2d(
                in_channels=original_module.in_channels,
                out_channels=original_module.out_channels,
                kernel_size=original_module.kernel_size,
                stride=original_module.stride,
                padding=original_module.padding,
                dilation=original_module.dilation,
                groups=original_module.groups,
                bias=use_bias,
                padding_mode=original_module.padding_mode,
            )

            copy_weights(original_module.weight, new_module.weight)
            if use_bias:
                copy_weights(original_module.bias, new_module.bias)
        elif config.get("name") == "fp16":
            new_module = original_module.half()

    elif mase_op in ("conv1d"):
        use_bias = original_module.bias is not None

        if config.get("name") == "int":
            if "input" in config:
                quant_desc_input = QuantDescriptor(
                    calib_method=config["input"]["calibrator"],
                    axis=config["input"]["quantize_axis"],
                    fake_quant=config["FakeQuantize"],
                )
                qnn.QuantConv1d.set_default_quant_desc_input(quant_desc_input)
            if "weight" in config:
                quant_desc_weight = QuantDescriptor(
                    calib_method=config["weight"]["calibrator"],
                    axis=config["weight"]["quantize_axis"],
                    fake_quant=config["FakeQuantize"],
                )
                qnn.QuantConv1d.set_default_quant_desc_weight(quant_desc_weight)

            new_module = qnn.QuantConv1d(
                in_channels=original_module.in_channels,
                out_channels=original_module.out_channels,
                kernel_size=original_module.kernel_size,
                stride=original_module.stride,
                padding=original_module.padding,
                dilation=original_module.dilation,
                groups=original_module.groups,
                bias=use_bias,
                padding_mode=original_module.padding_mode,
            )

            copy_weights(original_module.weight, new_module.weight)
            if use_bias:
                copy_weights(original_module.bias, new_module.bias)

        elif config.get("name") == "fp16":
            new_module = original_module.to(torch.float16)

    # elif mase_op == "relu":
    #     new_module_cls = quantized_module_map[f"relu_{quant_name}"]
    #     new_module = new_module_cls(inplace=original_module.inplace, config=config)

    # elif mase_op == "avg_pool2d":
    #     new_module_cls = quantized_module_map[f"avg_pool2d_{quant_name}"]
    #     new_module = new_module_cls(
    #         kernel_size=original_module.kernel_size,
    #         stride=original_module.stride,
    #         padding=original_module.padding,
    #         ceil_mode=original_module.ceil_mode,
    #         count_include_pad=original_module.count_include_pad,
    #         divisor_override=original_module.divisor_override,
    #         config=config,
    #     )
    # elif mase_op == "adaptive_avg_pool2d":
    #     new_module_cls = quantized_module_map[f"adaptive_avg_pool2d_{quant_name}"]
    #     new_module = new_module_cls(
    #         output_size=original_module.output_size, config=config
    #     )
    else:
        raise NotImplementedError(
            f"Unsupported module class {original_module_cls} to modify"
        )
    return new_module


# def create_new_fn(node, config: Dict):
#     mase_op = node.meta["mase"].parameters["common"]["mase_op"]
#     quant_name = config.get("name")
#     func_name = f"{mase_op}_{quant_name}"
#     new_func = quantized_func_map[func_name]
#     args, kwargs = node.args, node.kwargs | {"config": config}
#     return new_func, args, kwargs
