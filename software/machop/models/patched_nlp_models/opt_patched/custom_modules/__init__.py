import os
from typing import Dict

import toml
import torch.nn as nn

from .....utils import copy_weights
from ..modeling_opt_patched import (
    OPTAttentionPatched,
    OPTDecoderLayerPatched,
    OPTDecoderPatched,
    OPTLearnedPositionalEmbedding,
)
from .integer import OPTAttentionInteger

__all__ = [
    "OPT_MODULE_CLS_MAP",
    "OPT_MODULE_CLS_NAME_TO_MODULE_CLS",
    "OPT_MODULE_CLS_TO_MODULE_CLS_NAME",
    "OPT_MODIFIED_MODULE_CLASSES",
    "opt_create_new_custom_module",
]

OPT_MODULE_CLS_MAP = {
    OPTAttentionPatched: {
        "integer": OPTAttentionInteger,
    }
}

OPT_MODULE_CLS_NAME_TO_MODULE_CLS = {
    "opt::learned_positional_embedding": OPTLearnedPositionalEmbedding,
    "opt::attention": OPTAttentionPatched,
    "opt::decoder_layer": OPTDecoderLayerPatched,
    "opt::decoder": OPTDecoderPatched,
}

OPT_MODULE_CLS_TO_MODULE_CLS_NAME = {
    c: n for n, c in OPT_MODULE_CLS_NAME_TO_MODULE_CLS.items()
}

OPT_MODIFIED_MODULE_CLASSES = []
for k, v in OPT_MODULE_CLS_MAP.items():
    for kk, vv in v.items():
        OPT_MODIFIED_MODULE_CLASSES.append(vv)


def opt_create_new_custom_module(original_module: nn.Module, config: Dict):
    """
    Return a new module given original module
    """
    original_module_cls = type(original_module)
    if isinstance(original_module, OPTAttentionPatched):
        new_module_cls = OPT_MODULE_CLS_MAP[original_module_cls][config["name"]]
        new_module = new_module_cls(
            embed_dim=original_module.embed_dim,
            num_heads=original_module.num_heads,
            dropout=original_module.dropout,
            is_decoder=original_module.is_decoder,
            bias=original_module.k_proj.bias is not None,
            config=config,
        )
        copy_weights(original_module.k_proj.weight, new_module.k_proj.weight)
        copy_weights(original_module.q_proj.weight, new_module.q_proj.weight)
        copy_weights(original_module.v_proj.weight, new_module.v_proj.weight)
        copy_weights(original_module.out_proj.weight, new_module.out_proj.weight)
        if original_module.k_proj.bias is not None:
            copy_weights(original_module.k_proj.bias, new_module.k_proj.bias)
            copy_weights(original_module.q_proj.bias, new_module.q_proj.bias)
            copy_weights(original_module.v_proj.bias, new_module.v_proj.bias)
            copy_weights(original_module.out_proj.bias, new_module.out_proj.bias)
    else:
        raise NotImplementedError(f"Unsupported model cls {original_module}")

    return new_module
