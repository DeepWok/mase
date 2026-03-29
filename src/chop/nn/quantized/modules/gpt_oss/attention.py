"""
Quantized GPT-OSS attention with RoPE, KV cache, softmax, and attention sink support.

GPT-OSS attention differs from LLaMA/Qwen3 in two ways:
  1. Learnable attention sinks: a per-head scalar logit appended to attention
     weights before softmax, then dropped after — acts as a learned "null" token.
  2. No q_norm/k_norm (unlike Qwen3).

The quantized forward reimplements eager attention with MX-format quantization
on QK/AV matmul inputs, RoPE, softmax, and KV cache.
"""

from typing import Optional, Tuple
from functools import partial

import torch
from torch import Tensor, nn, LongTensor
import torch.nn.functional as F
from transformers.models.gpt_oss.modeling_gpt_oss import (
    apply_rotary_pos_emb,
    repeat_kv,
    Cache,
    GptOssAttention,
)

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.functional.softmax import softmax_minifloat
from chop.nn.quantized.functional.kvcache import kv_cache_mxfp, kv_cache_mxint


def _eager_attention_forward_with_sinks_mxfp(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    qk_bypass=False,
    qk_config=None,
    av_bypass=False,
    av_config=None,
    softmax_bypass=False,
    softmax_config=None,
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
        attn_weights = attn_weights + causal_mask

    # GPT-OSS attention sinks: append per-head learnable logit, softmax, then drop
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(
        query.shape[0], -1, query.shape[-2], -1
    )
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

    if not softmax_bypass:
        combined_logits = softmax_minifloat(combined_logits, softmax_config, dim=-1)
    else:
        combined_logits = F.softmax(
            combined_logits, dim=-1, dtype=combined_logits.dtype
        )

    scores = combined_logits[..., :-1]  # drop the sink column

    scores = F.dropout(scores, p=dropout, training=module.training)

    if not av_bypass:
        a_quantizer = partial(
            mxfp_quantizer,
            block_size=av_config["data_in_block_size"],
            element_exp_bits=av_config["data_in_exponent_width"],
            element_frac_bits=av_config["data_in_frac_width"],
            block_dim=-1,
        )
        scores = a_quantizer(scores)

    attn_output = torch.matmul(scores, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, scores


def _eager_attention_forward_with_sinks_mxint(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    qk_bypass=False,
    qk_config=None,
    av_bypass=False,
    av_config=None,
    softmax_bypass=False,
    softmax_config=None,
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
        attn_weights = attn_weights + causal_mask

    # GPT-OSS attention sinks
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(
        query.shape[0], -1, query.shape[-2], -1
    )
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

    if not softmax_bypass:
        combined_logits = softmax_minifloat(combined_logits, softmax_config, dim=-1)
    else:
        combined_logits = F.softmax(
            combined_logits, dim=-1, dtype=combined_logits.dtype
        )

    scores = combined_logits[..., :-1]

    scores = F.dropout(scores, p=dropout, training=module.training)

    if not av_bypass:
        a_quantizer = partial(
            mxint_quantizer,
            block_size=av_config["data_in_block_size"],
            element_bits=av_config["data_in_width"],
            block_dim=-1,
        )
        scores = a_quantizer(scores)

    attn_output = torch.matmul(scores, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, scores


class GptOssAttentionMXFP(GptOssAttention):
    """MXFP-quantized GptOssAttention with RoPE, KV cache, softmax, and sink support."""

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx)
        q_config = q_config or {}
        self.qk_config = q_config.get("qk_matmul", {})
        self.av_config = q_config.get("av_matmul", {})
        self.rope_config = q_config.get("rope", {})
        self.softmax_config = q_config.get("softmax", {})
        self.kv_cache_config = q_config.get("kv_cache", {})
        self.qk_bypass = self.qk_config.get("bypass", False)
        self.av_bypass = self.av_config.get("bypass", False)
        self.rope_bypass = self.rope_config.get("bypass", False)
        self.softmax_bypass = self.softmax_config.get("bypass", False)
        self.kv_cache_bypass = self.kv_cache_config.get("bypass", False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                self.rope_config,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            if not self.kv_cache_bypass:
                key_states, value_states = kv_cache_mxfp(
                    key_states,
                    value_states,
                    self.kv_cache_config,
                )
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attn_output, attn_weights = _eager_attention_forward_with_sinks_mxfp(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            qk_bypass=self.qk_bypass,
            qk_config=self.qk_config,
            av_bypass=self.av_bypass,
            av_config=self.av_config,
            softmax_bypass=self.softmax_bypass,
            softmax_config=self.softmax_config,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GptOssAttentionMXInt(GptOssAttention):
    """MXInt-quantized GptOssAttention with RoPE, KV cache, softmax, and sink support."""

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx)
        q_config = q_config or {}
        self.qk_config = q_config.get("qk_matmul", {})
        self.av_config = q_config.get("av_matmul", {})
        self.rope_config = q_config.get("rope", {})
        self.softmax_config = q_config.get("softmax", {})
        self.kv_cache_config = q_config.get("kv_cache", {})
        self.qk_bypass = self.qk_config.get("bypass", False)
        self.av_bypass = self.av_config.get("bypass", False)
        self.rope_bypass = self.rope_config.get("bypass", False)
        self.softmax_bypass = self.softmax_config.get("bypass", False)
        self.kv_cache_bypass = self.kv_cache_config.get("bypass", False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                self.rope_config,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            if not self.kv_cache_bypass:
                key_states, value_states = kv_cache_mxint(
                    key_states,
                    value_states,
                    self.kv_cache_config,
                )
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attn_output, attn_weights = _eager_attention_forward_with_sinks_mxint(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            qk_bypass=self.qk_bypass,
            qk_config=self.qk_config,
            av_bypass=self.av_bypass,
            av_config=self.av_config,
            softmax_bypass=self.softmax_bypass,
            softmax_config=self.softmax_config,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
