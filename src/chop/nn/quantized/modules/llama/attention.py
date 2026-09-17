"""Llama attention quantisation modules with phase-aware runtime dispatch.

Phase-split contract (decode-side disaggregated serving, e.g. PLENA):
1. Runtime phase (``prefill`` / ``decode``) is written by decoder-layer
   pre-hooks installed by the quantize pass; attention only consumes it.
2. Every in-attention stage (qk / av / rope / softmax / kv_cache) resolves
   its config per phase. A decode-only deployment bypasses all prefill
   stages (FP prefill chip) while decode runs fully quantised.
3. KV-cache handoff: with explicit ``kv_cache_handoff="decode_format"``
   prefill KV writes use the DECODE bucket's kv_cache config — the prefill
   chip computes in FP but its KV lands in the decode chip's HBM in the
   decode chip's MX format. The default ``"fp"`` preserves prefill KV.
4. When all in-attention stages are bypassed for the current phase, the
   forward falls through to HF's configured backend (sdpa / flash / eager),
   so an FP prefill keeps full-speed attention.
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
    eager_attention_forward as _hf_eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from functools import partial

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.functional.softmax import softmax_minifloat
from chop.nn.quantized.functional.kvcache import (
    kv_cache_mx,
    kv_cache_mxfp,
    kv_cache_mxfp_rotate,
    kv_cache_mxint,
    kv_cache_mxint_rotate,
)
from chop.nn.quantized.functional.attention import (
    eager_attention_forward_mxfp as _eager_attention_forward_mxfp,
    eager_attention_forward_mxfp_rotate as _eager_attention_forward_mxfp_rotate,
    eager_attention_forward_mxint as _eager_attention_forward_mxint,
    eager_attention_forward_mxint_rotate as _eager_attention_forward_mxint_rotate,
)
from chop.nn.quantized.modules.phase_context import get_runtime_phase
from chop.nn.quantized.modules.phase_config import (
    ATTENTION_STAGES,
    normalize_phase_q_config,
    resolve_stage_config,
)

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
    so we shouldn't force eager. Mirrors HF Llama's own dispatch line.
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


class _PhaseAwareAttentionMixin:
    """Per-phase stage-config resolution shared by the MX attention classes.

    Stage configs are resolved once at ``__init__`` (not per forward): during
    autoregressive decoding the forward runs once per generated token, so
    per-call dict copies would be pure overhead.
    """

    _STAGES = ATTENTION_STAGES
    # Short keys used in the resolved dicts, aligned with ATTENTION_STAGES.
    _STAGE_KEYS = tuple(s.replace("_matmul", "") for s in ATTENTION_STAGES)

    def _init_phase_attention_config(self, q_config: dict) -> None:
        self.q_config = q_config or {}
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self._phase_stage_cfgs = {
            phase: self._resolve_phase_stage_cfgs(phase)
            for phase in ("prefill", "decode")
        }
        # Legacy flat attributes mirror the decode side (the quantised chip)
        # so existing tooling that inspects e.g. ``self.qk_config`` still sees
        # the operative quantisation settings.
        decode_cfgs = self._phase_stage_cfgs["decode"]
        self.qk_config = decode_cfgs["qk_config"]
        self.av_config = decode_cfgs["av_config"]
        self.rope_config = decode_cfgs["rope_config"]
        self.softmax_config = decode_cfgs["softmax_config"]
        self.kv_cache_config = decode_cfgs["kv_cache_config"]
        self.qk_bypass = decode_cfgs["qk_bypass"]
        self.av_bypass = decode_cfgs["av_bypass"]
        self.rope_bypass = decode_cfgs["rope_bypass"]
        self.softmax_bypass = decode_cfgs["softmax_bypass"]
        self.kv_cache_bypass = decode_cfgs["kv_cache_bypass"]

    def _resolve_phase_stage_cfgs(self, phase: str) -> dict:
        cfgs = {}
        for stage, key in zip(self._STAGES, self._STAGE_KEYS):
            stage_cfg = resolve_stage_config(self.phase_q_config, phase, stage)
            cfgs[f"{key}_config"] = stage_cfg
            cfgs[f"{key}_bypass"] = stage_cfg.get("bypass", False)
        return cfgs

    def _active_stage_cfgs(self) -> dict:
        return self._phase_stage_cfgs[get_runtime_phase()]

    def _apply_kv_cache_phase_aware(
        self,
        key_states,
        value_states,
        past_key_values,
        cache_kwargs,
        stage,
        kv_quantizer,
    ):
        """Update the KV cache with quantise-on-write handoff semantics.

        The quantised K/V go into the cache — that is the copy living in the
        decode chip's HBM. What the current forward computes attention over
        depends on the phase:

        - decode: the full cache as stored (quantised), including the newly
          appended entries — the decode chip reads K/V from its HBM.
        - prefill: the prompt's own K/V stay FP for this forward — the
          prefill chip computes attention on-chip at full precision and only
          the handoff copy is quantised.

        With ``kv_cache`` bypassed (or no cache) this is a plain update.
        """

        if past_key_values is None:
            if (
                get_runtime_phase() == "decode"
                and not stage["kv_cache_bypass"]
                and kv_quantizer is not None
            ):
                return kv_quantizer(
                    key_states,
                    value_states,
                    stage["kv_cache_config"],
                )
            return key_states, value_states
        if stage["kv_cache_bypass"] or kv_quantizer is None:
            return past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        k_store, v_store = kv_quantizer(
            key_states, value_states, stage["kv_cache_config"]
        )
        k_all, v_all = past_key_values.update(
            k_store, v_store, self.layer_idx, cache_kwargs
        )
        if get_runtime_phase() == "prefill":
            q_len = key_states.shape[-2]
            if k_all.shape[-2] == q_len:
                return key_states, value_states
            # Continued prefill: earlier positions come from the cache (in
            # handoff format), the current chunk stays FP.
            k_all = torch.cat([k_all[..., :-q_len, :], key_states], dim=-2)
            v_all = torch.cat([v_all[..., :-q_len, :], value_states], dim=-2)
        return k_all, v_all


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
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

    @classmethod
    def from_self(cls, attention: LlamaAttention, q_config: dict = None):
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
        # strict=False: LSQInteger quantizer submodules add params not in base LlamaAttention
        new_attn.load_state_dict(attention.state_dict(), strict=False)
        return new_attn


class LlamaAttentionMXFP(_PhaseAwareAttentionMixin, LlamaAttention):
    """MXFP-quantized LlamaAttention with per-phase stage dispatch."""

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
        # HF compatibility: tolerate the legacy cache kwarg alias.
        past_key_values = kwargs.pop("past_key_value", past_key_values)
        stage = self._active_stage_cfgs()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
            kv_cache_mx,
        )

        # Skip-replace: if no in-attention quant stage is active for this
        # phase, fall back to whatever attention backend HF was configured
        # for (sdpa / fa2 / eager). KV-cache / RoPE quant happens above and
        # is unaffected. An FP prefill therefore keeps full-speed attention.
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
    def from_self(cls, attention: LlamaAttention, q_config: dict = None):
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


class LlamaAttentionMXInt(_PhaseAwareAttentionMixin, LlamaAttention):
    """MXInt-quantized LlamaAttention with per-phase stage dispatch."""

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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
            kv_cache_mx,
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
    def from_self(cls, attention: LlamaAttention, q_config: dict = None):
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


class LlamaAttentionMXIntRotate(LlamaAttentionMXInt):
    """LlamaAttentionMXInt with online Hadamard rotation around the
    activation quantizers (KV-cache, Q@K, A@V). Mirrors
    ``Qwen3AttentionMXIntRotate``.

    Per-stage rotate toggles let the rotation search swap stages
    independently. Each defaults True (preserves "all rotated" baseline if
    the user doesn't pre-set them); the search drives them via the matching
    block in q_config (decode bucket for phase-structured configs — rotation
    is a property of the quantised decode chip):

        q_config["qk_matmul"]["rotate"]   = True | False  (default True)
        q_config["av_matmul"]["rotate"]   = True | False  (default True)
        q_config["kv_cache"]["rotate"]    = True | False  (default True)

    Explicit decode-format handoff uses the decode rotation decision for both
    prefill and decode cache writes.
    """

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx, q_config=q_config)
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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
        def kv_quantizer(key, value, config):
            return kv_cache_mx(
                key,
                value,
                config,
                rotate=self.kv_cache_use_rotate,
            )
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


class LlamaAttentionMXFPRotate(LlamaAttentionMXFP):
    """LlamaAttentionMXFP with online Hadamard rotation around the activation
    quantizers (KV-cache, Q@K, A@V). MXFP analogue of
    ``LlamaAttentionMXIntRotate``; per-stage rotate toggles keep the
    rotation-search machinery uniform across formats:

        q_config["qk_matmul"]["rotate"]   = True | False  (default True)
        q_config["av_matmul"]["rotate"]   = True | False  (default True)
        q_config["kv_cache"]["rotate"]    = True | False  (default True)
    """

    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config, layer_idx, q_config=q_config)
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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
        def kv_quantizer(key, value, config):
            return kv_cache_mx(
                key,
                value,
                config,
                rotate=self.kv_cache_use_rotate,
            )
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
                "MXFP-rotate-quantized attention requires _attn_implementation='eager' "
                "when any of qk/av/softmax stages are active."
            )
            attn_output, attn_weights = _eager_attention_forward_mxfp_rotate(
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
