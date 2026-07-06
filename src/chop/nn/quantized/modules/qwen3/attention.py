"""Qwen3 attention quantisation modules with phase-aware runtime dispatch.

Shares the phase-split contract of the Llama modules (see
``llama/attention.py``): decoder-layer pre-hooks write the runtime phase,
every in-attention stage resolves its config per phase, and the KV-cache
handoff rule lets an FP prefill still write KV into the decode chip's MX
format for disaggregated serving.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor, LongTensor
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    Cache,
    Qwen3Attention,
    eager_attention_forward as _hf_eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.functional.kvcache import (
    kv_cache_mxfp,
    kv_cache_mxint,
    kv_cache_mxint_rotate,
)
from chop.nn.quantized.functional.attention import (
    eager_attention_forward_mxfp as _eager_attention_forward_mxfp,
    eager_attention_forward_mxint as _eager_attention_forward_mxint,
    eager_attention_forward_mxint_rotate as _eager_attention_forward_mxint_rotate,
)
from chop.nn.quantized.modules.llama.attention import _PhaseAwareAttentionMixin

import logging

logger = logging.getLogger(__name__)


def _hf_attention_dispatch(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    **kwargs,
):
    """Call HF's configured attention backend (sdpa / flash / eager / ...).

    Used when all in-attention quant stages (qk / av / softmax) are bypassed —
    in that case the wrapper has nothing to inject inside the attention compute,
    so we shouldn't force eager. Mirrors HF Qwen3's own dispatch line.
    """
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, _hf_eager_attention_forward
    )
    return attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        **kwargs,
    )


class Qwen3AttentionMXFP(_PhaseAwareAttentionMixin, Qwen3Attention):
    """MXFP-quantized Qwen3Attention with per-phase stage dispatch."""

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx)
        self._init_phase_attention_config(q_config)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        past_key_values = kwargs.pop("past_key_value", past_key_values)
        stage = self._active_stage_cfgs()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen3-specific: q_norm/k_norm applied after projection, before transpose
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not stage["rope_bypass"]:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                stage["rope_config"],
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = self._apply_kv_cache_phase_aware(
            key_states,
            value_states,
            past_key_values,
            cache_kwargs,
            stage,
            kv_cache_mxfp,
        )

        if stage["qk_bypass"] and stage["av_bypass"] and stage["softmax_bypass"]:
            attn_output, attn_weights = _hf_attention_dispatch(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        else:
            assert self.config._attn_implementation == "eager", (
                "MXFP-quantized eager attention requires _attn_implementation='eager' "
                "when any of qk/av/softmax stages are active."
            )
            attn_output, attn_weights = _eager_attention_forward_mxfp(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=stage["qk_bypass"],
                qk_config=stage["qk_config"],
                av_bypass=stage["av_bypass"],
                av_config=stage["av_config"],
                softmax_bypass=stage["softmax_bypass"],
                softmax_config=stage["softmax_config"],
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_self(cls, attention: Qwen3Attention, q_config: dict = None):
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


class Qwen3AttentionMXInt(_PhaseAwareAttentionMixin, Qwen3Attention):
    """MXInt-quantized Qwen3Attention with per-phase stage dispatch."""

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx)
        self._init_phase_attention_config(q_config)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        past_key_values = kwargs.pop("past_key_value", past_key_values)
        stage = self._active_stage_cfgs()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen3-specific: q_norm/k_norm applied after projection, before transpose
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not stage["rope_bypass"]:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                stage["rope_config"],
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = self._apply_kv_cache_phase_aware(
            key_states,
            value_states,
            past_key_values,
            cache_kwargs,
            stage,
            kv_cache_mxint,
        )

        if stage["qk_bypass"] and stage["av_bypass"] and stage["softmax_bypass"]:
            attn_output, attn_weights = _hf_attention_dispatch(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        else:
            assert self.config._attn_implementation == "eager", (
                "MXInt-quantized eager attention requires _attn_implementation='eager' "
                "when any of qk/av/softmax stages are active."
            )
            attn_output, attn_weights = _eager_attention_forward_mxint(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=stage["qk_bypass"],
                qk_config=stage["qk_config"],
                av_bypass=stage["av_bypass"],
                av_config=stage["av_config"],
                softmax_bypass=stage["softmax_bypass"],
                softmax_config=stage["softmax_config"],
                **kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_self(cls, attention: Qwen3Attention, q_config: dict = None):
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


class Qwen3AttentionMXIntRotate(Qwen3AttentionMXInt):
    """Qwen3AttentionMXInt with online Hadamard rotation around the activation
    quantizers. Mirrors Plena's ``online_rotate=True`` semantics: the KV cache
    quantize and the Q/A activation quantizes inside the eager-attention
    forward all go through ``mxint_rotate_quantizer``.

    Per-stage toggles let the rotation search evaluate qk_matmul / av_matmul /
    kv_cache rotations independently within a single attention class. Each
    flag defaults to True so an unmodified rotate config keeps the original
    "all three rotated" behavior. Toggle via the matching block in q_config
    (decode bucket for phase-structured configs):

        q_config["qk_matmul"]["rotate"]   = True | False  (default True)
        q_config["av_matmul"]["rotate"]   = True | False  (default True)
        q_config["kv_cache"]["rotate"]    = True | False  (default True)

    The flags are shared across phases: a prefill KV handoff write in
    decode_format uses the same rotation decision as decode KV writes, so
    the cache stays in one consistent basis.

    Note on ``hadamard_dim``: the rotation runs along the last dim of each
    tensor — head_dim for Q (always supported) and seq_len for the A-side
    rotate (often falls back to plain MXINT with a warning when seq_len isn't
    in the Hadamard table).
    """

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx, q_config=q_config)
        # Per-stage rotate toggles (default True = original "all rotated"
        # behavior). The rotation search flips these per trial.
        decode_cfgs = self._phase_stage_cfgs["decode"]
        self.qk_use_rotate = decode_cfgs["qk_config"].get("rotate", True)
        self.av_use_rotate = decode_cfgs["av_config"].get("rotate", True)
        self.kv_cache_use_rotate = decode_cfgs["kv_cache_config"].get("rotate", True)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        past_key_values = kwargs.pop("past_key_value", past_key_values)
        stage = self._active_stage_cfgs()

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
        if not stage["rope_bypass"]:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                stage["rope_config"],
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        kv_quantizer = kv_cache_mxint_rotate if self.kv_cache_use_rotate else kv_cache_mxint
        key_states, value_states = self._apply_kv_cache_phase_aware(
            key_states,
            value_states,
            past_key_values,
            cache_kwargs,
            stage,
            kv_quantizer,
        )

        if stage["qk_bypass"] and stage["av_bypass"] and stage["softmax_bypass"]:
            attn_output, attn_weights = _hf_attention_dispatch(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        else:
            assert self.config._attn_implementation == "eager", (
                "MXInt-rotate-quantized attention requires _attn_implementation='eager' "
                "when any of qk/av/softmax stages are active."
            )
            attn_output, attn_weights = _eager_attention_forward_mxint_rotate(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=stage["qk_bypass"],
                qk_config=stage["qk_config"],
                av_bypass=stage["av_bypass"],
                av_config=stage["av_config"],
                softmax_bypass=stage["softmax_bypass"],
                softmax_config=stage["softmax_config"],
                qk_use_rotate=self.qk_use_rotate,
                av_use_rotate=self.av_use_rotate,
                **kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
