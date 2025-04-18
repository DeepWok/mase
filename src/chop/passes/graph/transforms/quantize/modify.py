"""
Some of the functions here are taken from the Modifier class we had before
"""

from typing import Dict

import torch
from chop.tools.utils import copy_weights, init_LinearLUT_weight, init_Conv2dLUT_weight
from torch import nn
import numpy as np

from chop.nn.quantized.functional import quantized_func_map
from chop.nn.quantized.modules import quantized_module_map

type_to_name_map = {
    nn.Linear: "linear",
    nn.Conv1d: "conv1d",
    nn.Conv2d: "conv2d",
}


def create_new_module(
    mase_op: str,
    original_module: nn.Module,
    config: dict,
    node_meta: dict,
    baseline_module: nn.Module = None,
    successor_module: nn.Module = None,
    input_layers=None,
    output_layers=None,
):
    original_module_cls = type(original_module)
    quant_name = config.get("name")

    if quant_name == "ternary":
        config.update(
            {"node_meta_stat": node_meta["mase"].parameters["software"]["args"]}
        )

    if mase_op == "linear":
        new_module_cls = quantized_module_map[f"linear_{quant_name}"]
        use_bias = original_module.bias is not None
        # NOTE: We don't support training with pruning on base module. Only quantized modules for now.
        use_pruning = any(
            isinstance(original_module, quantized_module)
            for quantized_module in quantized_module_map.values()
        ) and (original_module.pruning_masks is not None)
        new_module = new_module_cls(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=use_bias,
            config=config,
        )
        if quant_name == "lutnet":
            initialized_weight, pruning_masks = init_LinearLUT_weight(
                levels=new_module.levels,
                k=new_module.k,
                original_pruning_mask=original_module.pruning_masks,
                original_weight=original_module.weight,
                in_features=original_module.in_features,
                out_features=original_module.out_features,
                new_module=new_module,
            )
            copy_weights(original_module.gamma, new_module.trainer.gamma)
            copy_weights(original_module.means, new_module.means)
            copy_weights(initialized_weight, new_module.trainer.weight)
            copy_weights(pruning_masks, new_module.trainer.pruning_masks)
        elif quant_name == "binary_residual":
            copy_weights(original_module.gamma, new_module.gamma)
            copy_weights(original_module.means, new_module.means)
            if original_module.means is not None:
                copy_weights(original_module.means, new_module.means)
            if use_pruning:
                copy_weights(original_module.pruning_masks, new_module.pruning_masks)
        elif quant_name == "logicnets":
            # LogicNets will require the node itself along with the subsequent activation for the initialization of the associated LUT.
            # Therefore, the activation layer, referred to as successor_module, needs to be passed for the module's initialization.
            new_module = new_module_cls(
                in_features=original_module.in_features,
                out_features=original_module.out_features,
                bias=use_bias,
                config=config,
                activation_module=successor_module,
                input_layers=input_layers,
                output_layers=output_layers,
            )
            copy_weights(original_module.weight, new_module.weight)

            # for LogicNets, bias must be copied before the truth tables are calculated
            if use_bias:
                copy_weights(original_module.bias, new_module.bias)
            new_module.calculate_truth_tables()
        else:
            copy_weights(original_module.weight, new_module.weight)
            if use_pruning:
                copy_weights(original_module.pruning_masks, new_module.pruning_masks)
            if use_bias:
                copy_weights(original_module.bias, new_module.bias)
    elif mase_op in ("conv1d", "conv2d"):
        name = f"{mase_op}_{quant_name}"
        new_module_cls = quantized_module_map[name]
        use_bias = original_module.bias is not None
        # NOTE: We don't support training with pruning on base module. Only quantized modules for now.
        use_pruning = any(
            isinstance(original_module, quantized_module)
            for quantized_module in quantized_module_map.values()
        ) and (original_module.pruning_masks is not None)
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
        if quant_name == "lutnet":
            # TODO: Initialize the weight based on the trained binaried network
            initialized_weight, pruning_masks = init_Conv2dLUT_weight(
                levels=new_module.levels,
                k=new_module.k,
                original_pruning_mask=original_module.pruning_masks,
                original_weight=original_module.weight,
                out_channels=original_module.out_channels,
                in_channels=original_module.in_channels,
                kernel_size=original_module.kernel_size,
                new_module=new_module,
            )
            copy_weights(
                original_module.gamma, new_module.trainer.gamma
            )  # TODO: Not sure about this. The paper doesn't specify this part.
            copy_weights(initialized_weight, new_module.trainer.weight)
            copy_weights(pruning_masks, new_module.trainer.pruning_masks)
            copy_weights(original_module.means, new_module.means)
        elif quant_name == "binary_residual":
            copy_weights(original_module.gamma, new_module.gamma)
            copy_weights(original_module.weight, new_module.weight)
            if original_module.means is not None:
                copy_weights(original_module.means, new_module.means)
            if use_pruning:
                copy_weights(original_module.pruning_masks, new_module.pruning_masks)
        elif quant_name == "logicnets":
            copy_weights(original_module.weight, new_module.weight)
            new_module.calculate_truth_tables()
        else:
            # TODO: LUTNet convolution does not support bias at the moment
            copy_weights(original_module.weight, new_module.weight)
            if use_pruning:
                copy_weights(original_module.pruning_masks, new_module.pruning_masks)
            if use_bias:
                copy_weights(original_module.weight, new_module.weight)
    elif mase_op == "convtranspose2d":
        name = f"{mase_op}_{quant_name}"
        new_module_cls = quantized_module_map[name]
        use_bias = original_module.bias is not None
        new_module = new_module_cls(
            in_channels=original_module.in_channels,
            out_channels=original_module.out_channels,
            kernel_size=original_module.kernel_size,
            stride=original_module.stride,
            padding=original_module.padding,
            output_padding=original_module.output_padding,
            groups=original_module.groups,
            bias=use_bias,
            dilation=original_module.dilation,
            padding_mode=original_module.padding_mode,
            config=config,
        )
    elif mase_op == "relu":
        new_module_cls = quantized_module_map[f"relu_{quant_name}"]
        new_module = new_module_cls(inplace=original_module.inplace, config=config)
    elif mase_op == "selu":
        new_module_cls = quantized_module_map[f"selu_{quant_name}"]
        new_module = new_module_cls(inplace=original_module.inplace, config=config)
    elif mase_op == "tanh":
        new_module_cls = quantized_module_map[f"tanh_{quant_name}"]
        new_module = new_module_cls(config=config)
    elif mase_op == "gelu":
        new_module_cls = quantized_module_map[f"gelu_{quant_name}"]
        new_module = new_module_cls(inplace=original_module.inplace, config=config)
    elif mase_op == "softsign":
        new_module_cls = quantized_module_map[f"softsign_{quant_name}"]
        new_module = new_module_cls(inplace=original_module.inplace, config=config)
    elif mase_op == "softplus":
        new_module_cls = quantized_module_map[f"softplus_{quant_name}"]
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
    elif mase_op == "batch_norm2d":
        new_module_cls = quantized_module_map[f"batch_norm2d_{quant_name}"]
        new_module = new_module_cls(
            num_features=original_module.num_features,
            eps=original_module.eps,
            momentum=original_module.momentum,
            affine=original_module.affine,
            track_running_stats=original_module.track_running_stats,
            config=config,
        )
        if original_module.affine:
            copy_weights(original_module.weight, new_module.weight)
            copy_weights(original_module.bias, new_module.bias)
    elif mase_op == "layer_norm":
        new_module_cls = quantized_module_map[f"layer_norm_{quant_name}"]
        new_module = new_module_cls(
            normalized_shape=original_module.normalized_shape,
            eps=original_module.eps,
            elementwise_affine=original_module.elementwise_affine,
            bias=original_module.bias,
            config=config,
        )
    elif mase_op == "group_norm":
        new_module_cls = quantized_module_map[f"group_norm_{quant_name}"]
        new_module = new_module_cls(
            num_groups=original_module.num_groups,
            num_channels=original_module.num_channels,
            eps=original_module.eps,
            affine=original_module.affine,
            config=config,
        )
    elif mase_op == "instance_norm2d":
        new_module_cls = quantized_module_map[f"instance_norm2d_{quant_name}"]
        new_module = new_module_cls(
            num_features=original_module.num_features,
            eps=original_module.eps,
            momentum=original_module.momentum,
            affine=original_module.affine,
            track_running_stats=original_module.track_running_stats,
            config=config,
        )
    elif mase_op == "rms_norm":
        new_module_cls = quantized_module_map[f"rms_norm_{quant_name}"]
        new_module = new_module_cls(
            normalized_shape=original_module.normalized_shape,
            eps=original_module.eps,
            elementwise_affine=original_module.elementwise_affine,
            config=config,
        )
    elif mase_op == "grouped_query_attention":
        new_module_cls = quantized_module_map[f"grouped_query_attention_{quant_name}"]
        new_module = new_module_cls(
            embed_dim=original_module.embed_dim,
            num_heads=original_module.num_heads,
            num_kv_heads=original_module.num_kv_heads,
            bias=original_module.bias,
            config=config,
            floor=True,
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
