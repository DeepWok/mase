import torch.nn as nn
import chop as chop
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

from .layer_utils import LinearNoise, Conv2dNoise, QuantBatchNorm2d
import torch


logger = get_logger(__name__)
set_logging_verbosity("debug")

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


def vit_module_level_add_noise(model, q_config = {}):
    from chop.passes.graph.utils import deepsetattr
    for module in model.named_modules():
        config = parse_q_config(module, q_config)
        if config is None:
            continue
        if get_module_type(module) == "conv2d":
            ori_module = module[1]
            new_module = Conv2dNoise(
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
            new_module = LinearNoise(
                ori_module.in_features,
                ori_module.out_features,
                q_config=config,
            )
            new_module.weight = ori_module.weight
            new_module.bias = ori_module.bias
            logger.debug(f"Replacing module: {module[0]}")

            deepsetattr(model, module[0], new_module)
        elif get_module_type(module) == "batch_norm":
            ori_module = module[1]
            # new_module = nn.BatchNorm2d(
            #     ori_module.num_features,
            #     ori_module.eps,
            #     ori_module.momentum,
            #     ori_module.affine,
            #     ori_module.track_running_stats,
            # )
            new_module = QuantBatchNorm2d(
                ori_module.num_features,
                ori_module.eps,
                ori_module.momentum,
                ori_module.affine,
                ori_module.track_running_stats,
                q_config=config,
            )
            new_module.weight = ori_module.weight
            new_module.bias = ori_module.bias
            new_module.running_mean = ori_module.running_mean
            new_module.running_var = ori_module.running_var
            new_module.num_batches_tracked = ori_module.num_batches_tracked
            deepsetattr(model, module[0], new_module)
            logger.debug(f"Replacing module: {module[0]}")

    return model