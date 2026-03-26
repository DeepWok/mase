"""
Fused RMSNorm + Residual Module Transform Pass
===============================================

A module-level MASE pass that fuses residual addition + RMSNorm in transformer
decoder layers by replacing the layer's forward method with one that calls a
hand-written Triton kernel.

Unlike the graph-level FX pass (passes/graph/transforms/fused_rmsnorm/), this
pass does NOT require FX tracing. It walks model.named_modules(), finds
decoder layers containing RMSNorm, and patches their forward methods directly.
This makes it compatible with all HuggingFace models, including those with
dynamic control flow that breaks FX tracing (e.g., Llama, Mistral).

This is the same approach used by Liger-Kernel in production.

Usage within the MASE pipeline:

    from chop.passes.module.transforms.fused_ops import (
        fused_rmsnorm_residual_transform_pass,
    )

    model, info = fused_rmsnorm_residual_transform_pass(model, {
        "casting_mode": "llama",   # "llama", "gemma", or "none"
    })

Part 2 of the ADLS kernel-fusion-aware optimisation pipeline.

Author : ADLS Group (Software Stream)
Date   : March 2026
"""

import logging
import types
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

try:
    from chop.passes.graph.transforms.fused_rmsnorm.triton_fused_add_rmsnorm import (
        FusedAddRMSNorm,
        FusedAddRMSNormModule,
    )
except ImportError:
    try:
        from triton_fused_add_rmsnorm import FusedAddRMSNorm, FusedAddRMSNormModule
    except ImportError:
        raise ImportError(
            "Could not import FusedAddRMSNorm. Ensure triton_fused_add_rmsnorm.py "
            "is available either via chop or on the Python path."
        )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm detection
# ---------------------------------------------------------------------------
_RMSNORM_CLASS_NAMES = frozenset({
    "RMSNorm",
    "LlamaRMSNorm",
    "MistralRMSNorm",
    "GemmaRMSNorm",
    "Qwen2RMSNorm",
    "InternLMRMSNorm",
    "CohereLayerNorm",
})

_DECODER_LAYER_CLASS_NAMES = frozenset({
    "LlamaDecoderLayer",
    "MistralDecoderLayer",
    "GemmaDecoderLayer",
    "Qwen2DecoderLayer",
    "InternLMDecoderLayer",
})


def _is_rmsnorm(module: nn.Module) -> bool:
    return any(name in type(module).__name__ for name in _RMSNORM_CLASS_NAMES)


def _is_decoder_layer(module: nn.Module) -> bool:
    return any(name in type(module).__name__ for name in _DECODER_LAYER_CLASS_NAMES)


def _get_eps(norm_module: nn.Module) -> float:
    """Extract epsilon from an RMSNorm module."""
    return getattr(norm_module, "variance_epsilon",
                   getattr(norm_module, "eps", 1e-6))


def _get_offset(norm_module: nn.Module) -> float:
    """Gemma adds 1.0 to the weight; others use 0.0."""
    return 1.0 if "Gemma" in type(norm_module).__name__ else 0.0


# ---------------------------------------------------------------------------
# Layer patching
# ---------------------------------------------------------------------------
def _patch_decoder_layer(
    layer: nn.Module,
    casting_mode: str,
) -> bool:
    """
    Patch a single decoder layer to fuse post-attention add+RMSNorm.

    This is a standalone function (not inlined in a loop) to avoid the
    Python closure-in-a-loop bug where all layers would share the last
    layer's captured variables.

    Original flow:
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        hidden_states = self_attn(hidden_states)
        hidden_states = residual + hidden_states           ← separate add
        residual = hidden_states
        hidden_states = post_attention_layernorm(residual)  ← separate norm

    Fused flow:
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        hidden_states = self_attn(hidden_states)
        hidden_states, residual = FusedAddRMSNorm(          ← single kernel
            residual, hidden_states, weight, eps
        )

    Returns True if patching succeeded, False otherwise.
    """
    # Find post-attention RMSNorm
    post_norm = getattr(layer, "post_attention_layernorm", None)
    if post_norm is None or not _is_rmsnorm(post_norm):
        return False

    eps = _get_eps(post_norm)
    offset = _get_offset(post_norm)
    norm_weight = post_norm.weight

    def fused_forward(self, hidden_states, **kwargs):
        residual = hidden_states

        # 1. Pre-attention norm (unchanged)
        hidden_states = self.input_layernorm(hidden_states)

        # 2. Self attention
        attn_out = self.self_attn(hidden_states=hidden_states, **kwargs)
        if isinstance(attn_out, tuple):
            hidden_states = attn_out[0]
        else:
            hidden_states = attn_out

        # 3. FUSED: residual + attn_output → post_attention_layernorm
        hidden_states, residual = FusedAddRMSNorm.apply(
            residual, hidden_states,
            norm_weight, eps, offset, casting_mode,
        )

        # 4. MLP (unchanged)
        hidden_states = self.mlp(hidden_states)

        # 5. Second residual add (unfused — could be fused in a future pass)
        hidden_states = residual + hidden_states

        return hidden_states

    layer.forward = types.MethodType(fused_forward, layer)
    return True


# ---------------------------------------------------------------------------
# Public pass function (MASE convention: takes model, returns (model, {}))
# ---------------------------------------------------------------------------
def fused_rmsnorm_residual_transform_pass(
    network: nn.Module,
    pass_args: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply fused RMSNorm + residual transformation to a model.

    Walks model.named_modules(), finds decoder layers with RMSNorm,
    and patches their forward methods to use a fused Triton kernel.

    Parameters
    ----------
    network : nn.Module
        The model to transform. Can be any HuggingFace causal LM
        (LlamaForCausalLM, MistralForCausalLM, etc.) or any model
        containing decoder layers with a post_attention_layernorm attribute.
    pass_args : dict, optional
        Configuration:
            - casting_mode (str): "llama" (default), "gemma", or "none"

    Returns
    -------
    tuple
        (transformed_model, info_dict) following MASE pass convention.
        info_dict contains:
            - num_fused (int): number of layers patched
            - fused_layers (list[str]): names of patched layers
    """
    pass_args = pass_args or {}
    casting_mode = pass_args.get("casting_mode", "llama")

    fused_layers = []

    for name, module in network.named_modules():
        if not _is_decoder_layer(module):
            continue

        if _patch_decoder_layer(module, casting_mode):
            fused_layers.append(name)
            logger.info(f"Fused post-attn add+RMSNorm in {name}")

    if not fused_layers:
        logger.info(
            "fused_rmsnorm_residual_transform_pass: no decoder layers with "
            "RMSNorm found. Model unchanged."
        )

    info = {
        "num_fused": len(fused_layers),
        "fused_layers": fused_layers,
    }

    logger.info(
        f"fused_rmsnorm_residual_transform_pass: patched {len(fused_layers)} "
        f"decoder layers (casting_mode={casting_mode})"
    )

    return network, info
