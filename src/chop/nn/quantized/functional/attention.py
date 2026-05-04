"""Shared eager-attention forward helpers for quantized attention modules.

Each architecture (Qwen3 / Qwen3-MoE / Llama / GLM4-MoE / ...) used to define
its own copy of ``_eager_attention_forward_mxint``; bodies were identical
modulo trivial dtype-cast variations. This module hosts a single
implementation per format that all of them call into.

Functions:
    eager_attention_forward_mxfp         — MXFP-quantized eager attention
    eager_attention_forward_mxint        — MXINT-quantized eager attention
    eager_attention_forward_mxint_rotate — MXINT + online Hadamard rotation
    eager_attention_forward_mxfp_rotate  — MXFP  + online Hadamard rotation
        (Llama-only for now; per-stage flags identical to the MXINT variant)
"""

from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn

# ``repeat_kv`` is a generic utility re-implemented identically across
# transformers' model files; importing from llama is just a stable choice.
from transformers.models.llama.modeling_llama import repeat_kv

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from .softmax import softmax_minifloat


def eager_attention_forward_mxfp(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    qk_bypass: bool = False,
    qk_config: dict = None,
    av_bypass: bool = False,
    av_config: dict = None,
    softmax_bypass: bool = False,
    softmax_config: dict = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if not qk_bypass:
        q_quantizer = partial(
            mxfp_quantizer,
            block_size=qk_config["data_in_block_size"],
            element_exp_bits=qk_config["data_in_exponent_width"],
            element_frac_bits=qk_config["data_in_frac_width"],
            block_dim=-1,
        )
        query = q_quantizer(query)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)

    if not softmax_bypass:
        attn_weights = softmax_minifloat(attn_weights, softmax_config, dim=-1)
    else:
        attn_weights = nn.functional.softmax(
            attn_weights.to(torch.float32),
            dim=-1,
        ).to(attn_weights.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )

    if not av_bypass:
        a_quantizer = partial(
            mxfp_quantizer,
            block_size=av_config["data_in_block_size"],
            element_exp_bits=av_config["data_in_exponent_width"],
            element_frac_bits=av_config["data_in_frac_width"],
            block_dim=-1,
        )
        attn_weights = a_quantizer(attn_weights)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def eager_attention_forward_mxint(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    qk_bypass: bool = False,
    qk_config: dict = None,
    av_bypass: bool = False,
    av_config: dict = None,
    softmax_bypass: bool = False,
    softmax_config: dict = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if not qk_bypass:
        q_quantizer = partial(
            mxint_quantizer,
            block_size=qk_config["data_in_block_size"],
            element_bits=qk_config["data_in_width"],
            block_dim=-1,
        )
        query = q_quantizer(query)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)

    if not softmax_bypass:
        attn_weights = softmax_minifloat(attn_weights, softmax_config, dim=-1)
    else:
        attn_weights = nn.functional.softmax(
            attn_weights.to(torch.float32),
            dim=-1,
        ).to(attn_weights.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )

    if not av_bypass:
        a_quantizer = partial(
            mxint_quantizer,
            block_size=av_config["data_in_block_size"],
            element_bits=av_config["data_in_width"],
            block_dim=-1,
        )
        attn_weights = a_quantizer(attn_weights)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def eager_attention_forward_mxfp_rotate(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    qk_bypass: bool = False,
    qk_config: dict = None,
    av_bypass: bool = False,
    av_config: dict = None,
    softmax_bypass: bool = False,
    softmax_config: dict = None,
    qk_use_rotate: bool = True,
    av_use_rotate: bool = True,
    **kwargs,
):
    """MXFP eager attention with online Hadamard rotation around the Q-side
    and A-side activation quantizers. MXFP equivalent of
    ``eager_attention_forward_mxint_rotate``; per-stage rotate toggles let
    the rotation search swap stages independently. When False, the stage
    falls back to plain ``mxfp_quantizer`` (no rotation).
    """
    from chop.nn.quantizers.rotation import mxfp_rotate_quantizer

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if not qk_bypass:
        if qk_use_rotate:
            query = mxfp_rotate_quantizer(
                query,
                hadamard_dim=query.shape[-1],
                block_size=qk_config["data_in_block_size"],
                element_exp_bits=qk_config["data_in_exponent_width"],
                element_frac_bits=qk_config["data_in_frac_width"],
                block_dim=-1,
                quantile_search=qk_config.get("clip_search", False),
                force_fp32=qk_config.get("force_fp32_had", False),
            )
        else:
            query = mxfp_quantizer(
                query,
                block_size=qk_config["data_in_block_size"],
                element_exp_bits=qk_config["data_in_exponent_width"],
                element_frac_bits=qk_config["data_in_frac_width"],
                block_dim=-1,
            )

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)

    if not softmax_bypass:
        attn_weights = softmax_minifloat(attn_weights, softmax_config, dim=-1)
    else:
        attn_weights = nn.functional.softmax(
            attn_weights.to(torch.float32),
            dim=-1,
        ).to(attn_weights.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )

    if not av_bypass:
        if av_use_rotate:
            attn_weights = mxfp_rotate_quantizer(
                attn_weights,
                hadamard_dim=attn_weights.shape[-1],
                block_size=av_config["data_in_block_size"],
                element_exp_bits=av_config["data_in_exponent_width"],
                element_frac_bits=av_config["data_in_frac_width"],
                block_dim=-1,
                quantile_search=av_config.get("clip_search", False),
                force_fp32=av_config.get("force_fp32_had", False),
            )
        else:
            attn_weights = mxfp_quantizer(
                attn_weights,
                block_size=av_config["data_in_block_size"],
                element_exp_bits=av_config["data_in_exponent_width"],
                element_frac_bits=av_config["data_in_frac_width"],
                block_dim=-1,
            )

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def eager_attention_forward_mxint_rotate(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    qk_bypass: bool = False,
    qk_config: dict = None,
    av_bypass: bool = False,
    av_config: dict = None,
    softmax_bypass: bool = False,
    softmax_config: dict = None,
    qk_use_rotate: bool = True,
    av_use_rotate: bool = True,
    **kwargs,
):
    """MXINT eager attention with online Hadamard rotation around the Q-side
    and A-side activation quantizers.

    Optional config keys per stage: ``clip_search``, ``force_fp32_had``.

    Per-stage rotate toggles (``qk_use_rotate`` / ``av_use_rotate``) let the
    caller mix rotated and non-rotated stages within the same attention
    instance. When False (and the corresponding bypass flag is also False),
    the stage falls back to plain ``mxint_quantizer``. Used by the rotation
    search to evaluate qk_matmul vs av_matmul independently.

    Note: ``hadamard_dim`` follows ``tensor.shape[-1]`` — this is head_dim for
    Q (always supported) but seq_len for the A side, which may not be in the
    Hadamard table; in that case ``mxint_rotate_quantizer`` falls back to plain
    MXINT with a one-time warning.
    """
    # Lazy import keeps fast_hadamard_transform optional for users that never
    # exercise the rotate path.
    from chop.nn.quantizers.rotation import mxint_rotate_quantizer

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if not qk_bypass:
        if qk_use_rotate:
            query = mxint_rotate_quantizer(
                query,
                hadamard_dim=query.shape[-1],
                block_size=qk_config["data_in_block_size"],
                element_bits=qk_config["data_in_width"],
                block_dim=-1,
                quantile_search=qk_config.get("clip_search", False),
                force_fp32=qk_config.get("force_fp32_had", False),
            )
        else:
            query = mxint_quantizer(
                query,
                block_size=qk_config["data_in_block_size"],
                element_bits=qk_config["data_in_width"],
                block_dim=-1,
            )

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)

    if not softmax_bypass:
        attn_weights = softmax_minifloat(attn_weights, softmax_config, dim=-1)
    else:
        attn_weights = nn.functional.softmax(
            attn_weights.to(torch.float32),
            dim=-1,
        ).to(attn_weights.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )

    if not av_bypass:
        if av_use_rotate:
            attn_weights = mxint_rotate_quantizer(
                attn_weights,
                hadamard_dim=attn_weights.shape[-1],
                block_size=av_config["data_in_block_size"],
                element_bits=av_config["data_in_width"],
                block_dim=-1,
                quantile_search=av_config.get("clip_search", False),
                force_fp32=av_config.get("force_fp32_had", False),
            )
        else:
            attn_weights = mxint_quantizer(
                attn_weights,
                block_size=av_config["data_in_block_size"],
                element_bits=av_config["data_in_width"],
                block_dim=-1,
            )

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
