from typing import Optional, Tuple

import torch
from torch import Tensor, nn, LongTensor
from transformers.models.glm4_moe.modeling_glm4_moe import (
    apply_rotary_pos_emb,
    Glm4MoeAttention,
    repeat_kv,
)
from transformers.cache_utils import Cache

from functools import partial

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.functional.softmax import softmax_minifloat
from chop.nn.quantized.functional.kvcache import kv_cache_mxfp, kv_cache_mxint

import logging

logger = logging.getLogger(__name__)


class Glm4MoeAttentionMXFP(Glm4MoeAttention):
    """MXFP-quantized Glm4MoeAttention.

    Handles optional q_norm/k_norm (applied before transpose, like Qwen3)
    and GLM4 MoE's partial RoPE.
    """

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

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        # GLM4 MoE: optional QK norm before transpose
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(
                query_states, key_states, cos, sin, self.rope_config,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not self.kv_cache_bypass:
                key_states, value_states = kv_cache_mxfp(
                    key_states, value_states, self.kv_cache_config,
                )
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
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

    @classmethod
    def from_attention(cls, attention: Glm4MoeAttention, q_config: dict = None):
        """Create a quantized Glm4MoeAttention that REUSES the original's
        child linears (q_proj, k_proj, v_proj, o_proj) and RMSNorms.

        This matters for tensor-parallel models: if the original attention's
        q_proj/k_proj/v_proj/o_proj have DTensor-backed weights (sharded by
        HF's tp_plan), instantiating a fresh ``cls(config, layer_idx)`` would
        allocate brand-new full-size weight tensors and lose the sharding.
        By sharing the child modules directly, the DTensor weights are
        preserved and the TP dispatch keeps working.
        """
        new_attn = cls.__new__(cls)
        nn.Module.__init__(new_attn)

        # Copy base Glm4MoeAttention state (matches the transformers __init__)
        new_attn.config = attention.config
        new_attn.layer_idx = attention.layer_idx
        new_attn.head_dim = attention.head_dim
        new_attn.num_key_value_groups = attention.num_key_value_groups
        new_attn.scaling = attention.scaling
        new_attn.rope_scaling = attention.rope_scaling
        new_attn.attention_dropout = attention.attention_dropout
        new_attn.is_causal = attention.is_causal

        # Reuse child submodules (preserves any DTensor-sharded weights)
        new_attn.q_proj = attention.q_proj
        new_attn.k_proj = attention.k_proj
        new_attn.v_proj = attention.v_proj
        new_attn.o_proj = attention.o_proj
        new_attn.use_qk_norm = attention.use_qk_norm
        if attention.use_qk_norm:
            new_attn.q_norm = attention.q_norm
            new_attn.k_norm = attention.k_norm

        # MX quant config (same as __init__)
        q_config = q_config or {}
        new_attn.qk_config = q_config.get("qk_matmul", {})
        new_attn.av_config = q_config.get("av_matmul", {})
        new_attn.rope_config = q_config.get("rope", {})
        new_attn.softmax_config = q_config.get("softmax", {})
        new_attn.kv_cache_config = q_config.get("kv_cache", {})
        new_attn.qk_bypass = new_attn.qk_config.get("bypass", False)
        new_attn.av_bypass = new_attn.av_config.get("bypass", False)
        new_attn.rope_bypass = new_attn.rope_config.get("bypass", False)
        new_attn.softmax_bypass = new_attn.softmax_config.get("bypass", False)
        new_attn.kv_cache_bypass = new_attn.kv_cache_config.get("bypass", False)

        return new_attn


class Glm4MoeAttentionMXInt(Glm4MoeAttention):
    """MXInt-quantized Glm4MoeAttention."""

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

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(
                query_states, key_states, cos, sin, self.rope_config,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not self.kv_cache_bypass:
                key_states, value_states = kv_cache_mxint(
                    key_states, value_states, self.kv_cache_config,
                )
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
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

    @classmethod
    def from_attention(cls, attention: Glm4MoeAttention, q_config: dict = None):
        """Create a quantized Glm4MoeAttention that REUSES the original's
        child linears and RMSNorms. See Glm4MoeAttentionMXFP.from_attention
        for why this matters for tensor-parallel models.
        """
        new_attn = cls.__new__(cls)
        nn.Module.__init__(new_attn)

        new_attn.config = attention.config
        new_attn.layer_idx = attention.layer_idx
        new_attn.head_dim = attention.head_dim
        new_attn.num_key_value_groups = attention.num_key_value_groups
        new_attn.scaling = attention.scaling
        new_attn.rope_scaling = attention.rope_scaling
        new_attn.attention_dropout = attention.attention_dropout
        new_attn.is_causal = attention.is_causal

        new_attn.q_proj = attention.q_proj
        new_attn.k_proj = attention.k_proj
        new_attn.v_proj = attention.v_proj
        new_attn.o_proj = attention.o_proj
        new_attn.use_qk_norm = attention.use_qk_norm
        if attention.use_qk_norm:
            new_attn.q_norm = attention.q_norm
            new_attn.k_norm = attention.k_norm

        q_config = q_config or {}
        new_attn.qk_config = q_config.get("qk_matmul", {})
        new_attn.av_config = q_config.get("av_matmul", {})
        new_attn.rope_config = q_config.get("rope", {})
        new_attn.softmax_config = q_config.get("softmax", {})
        new_attn.kv_cache_config = q_config.get("kv_cache", {})
        new_attn.qk_bypass = new_attn.qk_config.get("bypass", False)
        new_attn.av_bypass = new_attn.av_config.get("bypass", False)
        new_attn.rope_bypass = new_attn.rope_config.get("bypass", False)
        new_attn.softmax_bypass = new_attn.softmax_config.get("bypass", False)
        new_attn.kv_cache_bypass = new_attn.kv_cache_config.get("bypass", False)

        return new_attn


def _eager_attention_forward_mxfp(
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

    if not softmax_bypass:
        attn_weights = softmax_minifloat(attn_weights, softmax_config, dim=-1)
    else:
        attn_weights = nn.functional.softmax(
            attn_weights.to(torch.float32), dim=-1,
        ).to(attn_weights.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training,
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


def _eager_attention_forward_mxint(
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

    if not softmax_bypass:
        attn_weights = softmax_minifloat(attn_weights, softmax_config, dim=-1)
    else:
        attn_weights = nn.functional.softmax(
            attn_weights.to(torch.float32), dim=-1,
        ).to(attn_weights.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training,
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
