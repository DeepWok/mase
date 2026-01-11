import torch.nn as nn
import logging

from chop.nn.pim.pim_layer import PIMLinear, PIMConv2d, LoraPIMLinear
from chop.tools import deepsetattr

logger = logging.getLogger(__name__)


def get_module_type(module):
    """
    Categorize a module into a predefined type for PIM transformation.

    :param module: A tuple containing (module_name, module_instance).
    :type module: tuple
    :return: The category name of the module (e.g., 'linear', 'conv2d', 'layer_norm', etc.) or None if not recognized.
    :rtype: str or None
    """
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
    """
    Parse the PIM configuration for a specific module based on its name or type.

    :param module: A tuple containing (module_name, module_instance).
    :type module: tuple
    :param q_config: The global PIM configuration dictionary.
    :type q_config: dict
    :return: The specific configuration dictionary for the module, or None if no match is found.
    :rtype: dict or None
    :raises ValueError: If the "by" key in q_config is invalid.
    """
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


def pim_matmul_transform_pass(model, q_config={}, lora_config=None):
    """
    Apply PIM (Process-in-Memory) transformation to the given nn.Module.

    This pass replaces supported layers (Linear, Conv2d) with their PIM-aware counterparts
    (PIMLinear, PIMConv2d) or LoRA-enabled PIM layers (LoraPIMLinear).

    :param model: The input network to be transformed.
    :type model: torch.nn.Module
    :param q_config: Configuration for the PIM transformation, specifying how to match modules and their parameters.
    :type q_config: dict, optional
    :param lora_config: Configuration for LoRA if applying LoRA-enabled PIM transformation.
    :type lora_config: dict, optional

    Example q_config:

    .. code-block:: python

        q_config = {
            "by": "type",
            "linear": {
                "config": {
                    "tile_type": "pcm",
                    "core_size": 256,
                    "num_bits": 8,
                    "programming_noise": True,
                    "read_noise": True,
                    "ir_drop": True,
                    "out_noise": True,
                }
            },
        }

    :return: A tuple containing the transformed model and an empty dictionary (for consistency with other passes).
    :rtype: tuple
    """
    for module in model.named_modules():
        config = parse_q_config(module, q_config)
        if config is None:
            continue
        if get_module_type(module) == "conv2d":
            ori_module = module[1]
            new_module = PIMConv2d(
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
            if lora_config is not None:
                new_module = LoraPIMLinear(
                    ori_module.in_features,
                    ori_module.out_features,
                    q_config=config,
                    lora_config=lora_config,
                )
            else:
                new_module = PIMLinear(
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
