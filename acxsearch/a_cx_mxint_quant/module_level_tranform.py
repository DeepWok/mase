import torch.nn as nn
import chop as chop
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

from .attention import MXIntAttention
from .linear import MXIntLinear
from .layer_norm import MXIntLayerNorm
from .gelu import MXIntGELU
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


def vit_module_level_quantize(model, q_config = {}):
    from chop.passes.graph.utils import deepsetattr
    for module in model.named_modules():
        config = parse_q_config(module, q_config)
        if config is None:
            continue
        if get_module_type(module) == "attention":
            ori_module = module[1]
            new_module = MXIntAttention(
                ori_module.head_dim * ori_module.num_heads,
                ori_module.num_heads,
                qkv_bias=True,
                q_config=config,
            )
            logger.debug(f"Replacing module: {module[0]}")
            dim = ori_module.head_dim * ori_module.num_heads

            if hasattr(ori_module, 'qkv') and ori_module.qkv is not None:
                qkv_weight = ori_module.qkv.weight.reshape(3, dim, dim)
                new_module.query.weight = nn.Parameter(qkv_weight[0])
                new_module.key.weight = nn.Parameter(qkv_weight[1])
                new_module.value.weight = nn.Parameter(qkv_weight[2])
                has_bias = False if ori_module.qkv.bias == None else True
                if has_bias:
                    qkv_bias = ori_module.qkv.bias.reshape(3, 1, dim)
                    new_module.query.bias = nn.Parameter(qkv_bias[0])
                    new_module.key.bias = nn.Parameter(qkv_bias[1])
                    new_module.value.bias = nn.Parameter(qkv_bias[2])
            else:
                new_module.query.weight = nn.Parameter(ori_module.query.weight)
                new_module.key.weight = nn.Parameter(ori_module.key.weight)
                new_module.value.weight = nn.Parameter(ori_module.value.weight)
                has_bias = False if ori_module.query.bias == None else True
                if has_bias:
                    new_module.query.bias = nn.Parameter(ori_module.query.bias)
                    new_module.key.bias = nn.Parameter(ori_module.key.bias)
                    new_module.value.bias = nn.Parameter(ori_module.value.bias)


            new_module.proj.weight = ori_module.proj.weight
            new_module.proj.bias = ori_module.proj.bias
            deepsetattr(model, module[0], new_module)
        elif get_module_type(module) == "layer_norm":
            ori_module = module[1]
            if ori_module.bias is not None:
                bias = True
            new_module = MXIntLayerNorm(
                ori_module.normalized_shape,
                eps=ori_module.eps,
                elementwise_affine=ori_module.elementwise_affine,
                bias=bias,
                q_config=config,
            )
            new_module.weight = ori_module.weight
            new_module.bias = ori_module.bias
            logger.debug(f"Replacing module: {module[0]}")

            deepsetattr(model, module[0], new_module)
        elif get_module_type(module) == "linear":
            if "attention" in module[0]:
                continue
            if module[0] == "head":
                continue
            ori_module = module[1]
            new_module = MXIntLinear(
                ori_module.in_features,
                ori_module.out_features,
                q_config=config,
            )
            new_module.weight = ori_module.weight
            new_module.bias = ori_module.bias
            logger.debug(f"Replacing linear module: {module[0]}")
            deepsetattr(model, module[0], new_module)
        elif get_module_type(module) == "gelu":
            ori_module = module[1]
            new_module = MXIntGELU(
                q_config=config,
            )
            logger.debug(f"Replacing module: {module[0]}")
            deepsetattr(model, module[0], new_module)
    return model