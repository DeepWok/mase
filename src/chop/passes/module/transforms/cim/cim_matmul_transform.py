import torch.nn as nn
import logging

from chop.nn.cim.cim_layer import CIMLinear, CIMConv2d, LoraCIMLinear
from chop.tools import deepsetattr

logger = logging.getLogger(__name__)


def get_module_type(module):
    class_name = module[1].__class__.__name__
    if "Linear" in class_name:
        return "linear"
    elif "LayerNorm" in class_name:
        return "layer_norm"
    elif "Attention" in class_name:
        if "Head" not in class_name:
            return "attention"
        else:
            return "attention_head"
    elif "GELU" in class_name:
        return "gelu"
    elif "Conv2d" in class_name:
        return "conv2d"
    elif "ReLU" in class_name:
        return "relu"
    elif "BatchNorm2d" in class_name:
        return "batch_norm"
    else:
        return None


def parse_q_config(module, q_config):
    if q_config.get("by") == "name":
        if module[0] in q_config:
            return q_config[module[0]]["config"]
        else:
            return None
    elif q_config.get("by") == "type":
        module_type = get_module_type(module)
        if module_type in q_config:
            return q_config[module_type]["config"]
        else:
            return None
    else:
        raise ValueError(f"Invalid q_config: {q_config}")


def cim_matmul_transform_pass(model, q_config={}, lora_config={}):
    for module in model.named_modules():
        config = parse_q_config(module, q_config)
        if config is None:
            continue
        if get_module_type(module) == "conv2d":
            ori_module = module[1]
            new_module = CIMConv2d(
                ori_module.in_channels,
                ori_module.out_channels,
                ori_module.kernel_size,
                ori_module.stride,
                ori_module.padding,
                q_config=config,
            )
            new_module.weight = ori_module.weight
            new_module.bias = ori_module.bias
            deepsetattr(model, module[0], new_module)
            logger.debug(f"Replacing module: {module[0]}")
        elif get_module_type(module) == "linear":
            ori_module = module[1]
            if lora_config is not {} or lora_config is not None:
                new_module = LoraCIMLinear(
                    ori_module.in_features,
                    ori_module.out_features,
                    q_config=config,
                    lora_config=lora_config,
                )
            else:
                new_module = CIMLinear(
                    ori_module.in_features,
                    ori_module.out_features,
                    q_config=config,
                )
            new_module.weight.data = ori_module.weight.data
            new_module.bias = ori_module.bias
            logger.debug(f"Replacing module: {module[0]}")

            deepsetattr(model, module[0], new_module)
        else:
            continue

    return model, {}
