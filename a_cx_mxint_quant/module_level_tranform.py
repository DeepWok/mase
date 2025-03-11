import torch.nn as nn
import chop as chop
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

from .attention_head import _ViTSelfAttentionHeadBase
from .attention import MXIntAttention
from chop.models.vision.vit.vit import Attention

from .linear import MXIntLinear
# from .layer_norm import MXIntLayerNorm
# from .gelu import MXIntGELU
import torch


logger = get_logger(__name__)
set_logging_verbosity("debug")
class MXIntGELU(nn.Module):
    def __init__(self, q_config = {}):
        super().__init__()
        self.q_config = q_config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        return out

class MXIntLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        q_config=None,
    ) -> None:
        self.q_config = q_config
        super().__init__(normalized_shape, eps, elementwise_affine, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

def vit_module_level_quantize(model, model_config = {}, q_config = {}):
    def parse_q_config(module, q_config):
        if q_config.get("by") == "name":
            if module[0] in q_config:
                return False, q_config[module[0]]["config"]
            else:
                return True, None
        elif q_config.get("by") == "type":
            module_name = module[1].__class__.__name__
            if "Linear" in module_name:
                if any("linear" in key for key in q_config.keys()):
                    if "linear1" in module[0] and "linear1" in q_config:
                        return False, q_config["linear1"]["config"]
                    elif "linear2" in module[0] and "linear2" in q_config:
                        return False, q_config["linear2"]["config"]
                    else:
                        return False, q_config["linear"]["config"]
                else:
                    return True, None
            elif "layer_norm" in q_config and "LayerNorm" in module_name:
                return False, q_config["layer_norm"]["config"]
            elif "attention" in q_config and "Attention" in module_name:
                return False, q_config["attention"]["config"]
            elif "gelu" in q_config and "GELU" in module_name:
                return False, q_config["gelu"]["config"]
            else:
                return True, None
        else:
            raise ValueError(f"Invalid q_config: {q_config}")

    from chop.passes.graph.utils import deepsetattr
    for module in model.named_modules():
        skip, config = parse_q_config(module, q_config)
        if skip:
            continue
        if isinstance(module[1], Attention):
            ori_module = module[1]
            new_module = MXIntAttention(
                model_config["dim"],
                model_config["num_heads"],
                qkv_bias=True,
                q_config=config,
            )
            logger.info(f"Replacing module: {module[0]}")
            dim = ori_module.head_dim * ori_module.num_heads

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

            new_module.proj.weight = ori_module.proj.weight
            new_module.proj.bias = ori_module.proj.bias
            deepsetattr(model, module[0], new_module)
        elif isinstance(module[1], nn.LayerNorm):
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
            logger.info(f"Replacing module: {module[0]}")

            deepsetattr(model, module[0], new_module)
        elif isinstance(module[1], nn.Linear) or isinstance(module[1], MXIntLinear):
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
            logger.info(f"Replacing linear module: {module[0]}")
            deepsetattr(model, module[0], new_module)
        elif isinstance(module[1], nn.GELU):
            ori_module = module[1]
            new_module = MXIntGELU(
                q_config=config,
            )
            logger.info(f"Replacing module: {module[0]}")
            deepsetattr(model, module[0], new_module)
    return model