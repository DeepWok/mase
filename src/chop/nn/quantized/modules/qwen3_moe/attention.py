from typing import Optional, Tuple

import torch
from torch import Tensor, nn, LongTensor
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    apply_rotary_pos_emb,
    Cache,
    Qwen3MoeAttention,
)

from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.functional.softmax import softmax_minifloat
from chop.nn.quantized.functional.kvcache import kv_cache_mxfp, kv_cache_mxint
from chop.nn.quantized.modules.qwen3.attention import (
    _eager_attention_forward_mxfp,
    _eager_attention_forward_mxint,
)


class Qwen3MoeAttentionMXFP(Qwen3MoeAttention):
    """MXFP-quantized Qwen3MoeAttention with RoPE, KV cache, and softmax quantization."""

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

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
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
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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

        attn_output, attn_weights = _eager_attention_forward_mxfp(
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


class Qwen3MoeAttentionMXInt(Qwen3MoeAttention):
    """MXInt-quantized Qwen3MoeAttention with RoPE, KV cache, and softmax quantization."""

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

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
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
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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

        attn_output, attn_weights = _eager_attention_forward_mxint(
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
