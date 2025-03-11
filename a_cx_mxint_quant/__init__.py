from .module_level_tranform import vit_module_level_quantize
from .quantizers import mxint_hardware, mxint_quant_block

from .linear import MXIntLinear
from .attention import MXIntAttention
from .module_level_tranform import MXIntLayerNorm, MXIntGELU
from .modules import MXIntPatchEmbed, MXIntAddition
from mase_components import get_module_dependencies
VIT_CUSTOM_OPS = {
    "modules": {
        MXIntPatchEmbed: {
            "args": {
                "data_in": "data_in",
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_patch_embed",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_patch_embed"
            ),
        },
        MXIntAttention: {
            "args": {
                "data_in": "data_in",
                "dim": "config",
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
        MXIntLayerNorm: {
            "args": {
                "data_in": "data_in",
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_layernorm",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_layernorm"
            ),
        },
        MXIntGELU: {
            "args": {
                "data_in": "data_in",
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_gelu",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_gelu"
            ),
        },
        MXIntLinear: {
            "args": {
                "data_in": "data_in",
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_linear",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_linear"
            ),
        },
        MXIntAddition: {
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
    },
}