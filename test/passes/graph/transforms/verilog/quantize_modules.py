
import torch.nn as nn
from mase_components import get_module_dependencies
from chop.nn.quantized.modules.attention import _ViTAttentionBase

import chop as chop
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

logger = get_logger(__name__)
set_logging_verbosity("debug")
from chop.models.vision.vit.vit import Attention
import torch
class MxIntPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        q_config: dict = None,
        norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.q_config = q_config
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.norm = norm_layer(embed_dim)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.distill_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), self.distill_token.expand(x.size(0), -1, -1), x), dim=1)
        return x
class ViTAttentionMxInt(_ViTAttentionBase):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        q_config: dict = None,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop)
        self.q_config = q_config

class MxIntGELU(nn.GELU):
    def __init__(
        self,
        q_config,
    ) -> None:
        super().__init__()
        self.q_config = q_config

class MxIntGELU(nn.GELU):
    def __init__(
        self,
        q_config,
    ) -> None:
        super().__init__()
        self.q_config = q_config

class MxIntLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        q_config,
    ) -> None:
        super().__init__(in_features, out_features)
        self.q_config = q_config

class MxIntLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normlized_shape,
        q_config,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
    ) -> None:
        super().__init__(normlized_shape, eps, elementwise_affine, bias)
        self.q_config = q_config

class MxIntAddition(nn.Module):
    def __init__(
        self,
        q_config,
    ) -> None:
        super().__init__()
        self.q_config = q_config
    
    def forward(self, x, y):
        return x + y

VIT_CUSTOM_OPS = {
    "modules": {
        ViTAttentionMxInt: {
            "args": {
                "dim": "data_in",
                "num_heads": "config",
                "qkv_bias": "config",
                "qk_norm": None,
                "attn_drop": None,
                "proj_drop": None,
                "norm_layer": None,
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_vit_attention_wrap",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_vit_attention_wrap"
            ),
        },
        MxIntLayerNorm: {
            "args": {
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_layernorm",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_layernorm"
            ),
        },
        MxIntGELU: {
            "args": {
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_gelu",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_gelu"
            ),
        },
        MxIntLinear: {
            "args": {
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_linear",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_linear"
            ),
        },
        MxIntAddition: {
            "args": {
                "input_0": "data_in",
                "input_1": "data_in",
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_addition",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_addition"
            ),
        },
        MxIntPatchEmbed: {
            "args": {
                "input_0": "data_in",
                "input_1": "data_in",
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_linear",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_patch_embed"
            ),
        },
    },
}


def vit_module_level_quantize(model, model_config, q_config):
    from chop.passes.graph.utils import deepsetattr
    for module in model.named_modules():
        if isinstance(module[1], Attention):
            ori_module = module[1]
            new_module = ViTAttentionMxInt(
                model_config["dim"],
                model_config["num_heads"],
                qkv_bias=True,
                q_config=q_config["user_defined_module"],
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
            new_module = MxIntLayerNorm(
                ori_module.normalized_shape,
                eps=ori_module.eps,
                elementwise_affine=ori_module.elementwise_affine,
                bias=bias,
                q_config=q_config,
            )
            new_module.weight = ori_module.weight
            new_module.bias = ori_module.bias
            logger.info(f"Replacing module: {module[0]}")

            deepsetattr(model, module[0], new_module)
        elif isinstance(module[1], nn.Linear):
            if "attention" in module[0]:
                continue
            ori_module = module[1]
            new_module = MxIntLinear(
                ori_module.in_features,
                ori_module.out_features,
                q_config=q_config,
            )
            new_module.weight = ori_module.weight
            new_module.bias = ori_module.bias
            logger.info(f"Replacing module: {module[0]}")

            deepsetattr(model, module[0], new_module)

        elif isinstance(module[1], nn.LayerNorm):
            ori_module = module[1]
            if ori_module.bias is not None:
                bias = True
            new_module = MxIntLayerNorm(
                ori_module.normalized_shape,
                eps=ori_module.eps,
                elementwise_affine=ori_module.elementwise_affine,
                bias=bias,
                q_config=q_config["layer_norm"],
            )
            new_module.weight = ori_module.weight
            new_module.bias = ori_module.bias
            print(f"LayerNorm {module[0]} was replaced")
            logger.info(f"Replacing module: {module[0]}")

            deepsetattr(model, module[0], new_module)
        elif isinstance(module[1], nn.GELU):
            ori_module = module[1]
            new_module = MxIntGELU(
                q_config=q_config["gelu"],
            )
            logger.info(f"Replacing module: {module[0]}")
            deepsetattr(model, module[0], new_module)
    return model