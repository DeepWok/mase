"""Llama attention quantization modules with phase-aware runtime dispatch.

Compatibility boundaries:
1. Keep module replacement API stable for existing passes.
2. Accept phase-structured configs (`prefill`/`decode`) without forcing new
   caller-side changes.
3. Preserve default decode behavior (`decode_policy='fp_only'`) while allowing
   explicit opt-in decode quantization (`decode_policy='quantized'`).
4. Keep a single local eager attention path (no backend fallback dispatch).

Runtime phase is set by decoder-layer pre-hooks in quantize pass. Attention
consumes context phase to select per-phase quantization settings.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn, LongTensor
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaConfig,
    Cache,
    repeat_kv,
    LlamaAttention,
)

from functools import partial

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.functional.softmax import softmax_minifloat
from chop.nn.quantized.functional.kvcache import kv_cache_mxfp, kv_cache_mxint
from chop.nn.quantized.modules.phase_context import (
    get_runtime_phase,
)
from chop.nn.quantized.modules.phase_config import (
    get_phase_subconfig,
    normalize_phase_q_config,
)

import logging

logger = logging.getLogger(__name__)


def _apply_decode_fp_only_bypass(
    phase_subconfig: dict, decode_policy: str, runtime_phase: str
) -> dict:
    """Apply `fp_only` decode policy to a phase sub-config.

    Why this helper exists:
    - We keep phase policy handling explicit and centralized instead of
      duplicating `if decode/fp_only` branches for each attention sub-stage.
    """

    if runtime_phase == "decode" and decode_policy == "fp_only":
        cfg = dict(phase_subconfig)
        cfg["bypass"] = True
        return cfg
    return phase_subconfig


class LlamaAttentionLSQInteger(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, q_config: dict = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.query_quan = LSQInteger(level=q_config["level"], sym=True)
        self.key_quan = LSQInteger(level=q_config["level"], sym=True)
        self.value_quan = LSQInteger(level=q_config["level"], sym=True)
        self.o_quant = LSQInteger(level=q_config["level"], sym=True)
        self.attn_quan = LSQInteger(level=q_config["level"], sym=False)
        self.after_attn_quan = LSQInteger(level=q_config["level"], sym=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # HF compatibility: tolerate legacy alias when callers still pass it.
        past_key_value = kwargs.pop("past_key_value", past_key_values)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.query_quan(query_states)
        key_states = self.key_quan(key_states)
        value_states = self.value_quan(value_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )
        attn_weights = self.attn_quan(attn_weights)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self.after_attn_quan(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = self.o_quant(attn_output)

        return attn_output, attn_weights


class LlamaAttentionMXFP(LlamaAttention):
    """MXFP-quantized LlamaAttention."""

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx)
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]

    def _resolve_phase_attention_quant_config(self, runtime_phase: str) -> dict:
        """Resolve attention quant config for the current runtime phase."""

        phase_subconfig, decode_policy = get_phase_subconfig(
            self.phase_q_config, runtime_phase
        )
        qk_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("qk_matmul", {}), decode_policy, runtime_phase
        )
        av_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("av_matmul", {}), decode_policy, runtime_phase
        )
        rope_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("rope", {}), decode_policy, runtime_phase
        )
        softmax_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("softmax", {}), decode_policy, runtime_phase
        )
        kv_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("kv_cache", {}), decode_policy, runtime_phase
        )
        return {
            "qk_config": qk_cfg,
            "av_config": av_cfg,
            "rope_config": rope_cfg,
            "softmax_config": softmax_cfg,
            "kv_cache_config": kv_cfg,
            "qk_bypass": qk_cfg.get("bypass", False),
            "av_bypass": av_cfg.get("bypass", False),
            "rope_bypass": rope_cfg.get("bypass", False),
            "softmax_bypass": softmax_cfg.get("bypass", False),
            "kv_cache_bypass": kv_cfg.get("bypass", False),
        }

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        past_key_value = kwargs.pop("past_key_value", past_key_values)
        # Phase is written by decoder-layer pre-hooks before input_layernorm.
        runtime_phase = get_runtime_phase()
        phase_attention_config = self._resolve_phase_attention_quant_config(
            runtime_phase
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not phase_attention_config["rope_bypass"]:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                phase_attention_config["rope_config"],
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not phase_attention_config["kv_cache_bypass"]:
                key_states, value_states = kv_cache_mxfp(
                    key_states,
                    value_states,
                    phase_attention_config["kv_cache_config"],
                )
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # Keep a single eager path for maintainability: all-bypass still uses
        # this function but bypass flags short-circuit quantization stages.
        attn_output, attn_weights = _eager_attention_forward_mxfp(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            qk_bypass=phase_attention_config["qk_bypass"],
            qk_config=phase_attention_config["qk_config"],
            av_bypass=phase_attention_config["av_bypass"],
            av_config=phase_attention_config["av_config"],
            softmax_bypass=phase_attention_config["softmax_bypass"],
            softmax_config=phase_attention_config["softmax_config"],
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_attention(cls, attention: LlamaAttention, q_config: dict = None):
        new_attn = cls(
            config=attention.config,
            layer_idx=attention.layer_idx,
            q_config=q_config,
        )
        device, dtype = (
            next(attention.parameters()).device,
            next(attention.parameters()).dtype,
        )
        new_attn = new_attn.to(dtype=dtype, device=device)
        new_attn.load_state_dict(attention.state_dict(), strict=True)
        return new_attn


class LlamaAttentionMXInt(LlamaAttention):
    """MXInt-quantized LlamaAttention."""

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx)
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]

    def _resolve_phase_attention_quant_config(self, runtime_phase: str) -> dict:
        """Resolve attention quant config for the current runtime phase."""

        phase_subconfig, decode_policy = get_phase_subconfig(
            self.phase_q_config, runtime_phase
        )
        qk_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("qk_matmul", {}), decode_policy, runtime_phase
        )
        av_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("av_matmul", {}), decode_policy, runtime_phase
        )
        rope_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("rope", {}), decode_policy, runtime_phase
        )
        softmax_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("softmax", {}), decode_policy, runtime_phase
        )
        kv_cfg = _apply_decode_fp_only_bypass(
            phase_subconfig.get("kv_cache", {}), decode_policy, runtime_phase
        )
        return {
            "qk_config": qk_cfg,
            "av_config": av_cfg,
            "rope_config": rope_cfg,
            "softmax_config": softmax_cfg,
            "kv_cache_config": kv_cfg,
            "qk_bypass": qk_cfg.get("bypass", False),
            "av_bypass": av_cfg.get("bypass", False),
            "rope_bypass": rope_cfg.get("bypass", False),
            "softmax_bypass": softmax_cfg.get("bypass", False),
            "kv_cache_bypass": kv_cfg.get("bypass", False),
        }

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        past_key_value = kwargs.pop("past_key_value", past_key_values)
        # Phase is written by decoder-layer pre-hooks before input_layernorm.
        runtime_phase = get_runtime_phase()
        phase_attention_config = self._resolve_phase_attention_quant_config(
            runtime_phase
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not phase_attention_config["rope_bypass"]:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                phase_attention_config["rope_config"],
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not phase_attention_config["kv_cache_bypass"]:
                key_states, value_states = kv_cache_mxint(
                    key_states,
                    value_states,
                    phase_attention_config["kv_cache_config"],
                )
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # Keep a single eager path for maintainability: all-bypass still uses
        # this function but bypass flags short-circuit quantization stages.
        attn_output, attn_weights = _eager_attention_forward_mxint(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            qk_bypass=phase_attention_config["qk_bypass"],
            qk_config=phase_attention_config["qk_config"],
            av_bypass=phase_attention_config["av_bypass"],
            av_config=phase_attention_config["av_config"],
            softmax_bypass=phase_attention_config["softmax_bypass"],
            softmax_config=phase_attention_config["softmax_config"],
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_attention(cls, attention: LlamaAttention, q_config: dict = None):
        new_attn = cls(
            config=attention.config,
            layer_idx=attention.layer_idx,
            q_config=q_config,
        )
        device, dtype = (
            next(attention.parameters()).device,
            next(attention.parameters()).dtype,
        )
        new_attn = new_attn.to(dtype=dtype, device=device)
        new_attn.load_state_dict(attention.state_dict(), strict=True)
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


class LlamaAttentionMXFPRotate(LlamaAttentionMXFP):
    """Rotation-variant compatibility alias for MXFP attention.

    The phase-split migration intentionally keeps one phase-aware eager path in
    this file. We retain the rotate class symbol to preserve mainline import
    contracts (`llama/__init__.py`) and downstream module map wiring.
    """

    pass


class LlamaAttentionMXIntRotate(LlamaAttentionMXInt):
    """Rotation-variant compatibility alias for MXInt attention.

    Keeping this symbol avoids breaking existing registry/import paths while we
    focus this PR on phase-split decode policy behavior.
    """

    pass
