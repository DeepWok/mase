"""Phase-aware Qwen3-MoE attention quantisation."""

from __future__ import annotations

from typing import Optional

from torch import LongTensor, Tensor
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Cache,
    Qwen3MoeAttention,
    apply_rotary_pos_emb,
    eager_attention_forward as _hf_eager_attention_forward,
)

from chop.nn.quantized.functional.attention import (
    eager_attention_forward_mxfp as _eager_attention_forward_mxfp,
    eager_attention_forward_mxfp_rotate as _eager_attention_forward_mxfp_rotate,
    eager_attention_forward_mxint as _eager_attention_forward_mxint,
    eager_attention_forward_mxint_rotate as _eager_attention_forward_mxint_rotate,
)
from chop.nn.quantized.functional.kvcache import kv_cache_mx
from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.modules.llama.attention import _PhaseAwareAttentionMixin
from chop.nn.quantized.modules.qwen3.attention import _apply_qk_norm


def _hf_attention_dispatch(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    **kwargs,
):
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, _hf_eager_attention_forward
    )
    return interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        **kwargs,
    )


class _PhaseAwareQwen3MoeAttention(_PhaseAwareAttentionMixin, Qwen3MoeAttention):
    _mx_format = "mxint"
    _rotate = False

    def __init__(self, config, layer_idx, q_config: dict | None = None):
        super().__init__(config, layer_idx)
        self._init_phase_attention_config(q_config)
        for phase_cfg in self._phase_stage_cfgs.values():
            kv_cfg = phase_cfg["kv_cache_config"]
            if "key" in kv_cfg or "value" in kv_cfg:
                if (
                    set(kv_cfg) != {"key", "value"}
                    or kv_cfg["key"] != kv_cfg["value"]
                ):
                    raise ValueError(
                        "Qwen3-MoE requires identical K/V cache precision"
                    )
        if self._rotate:
            decode_cfg = self._phase_stage_cfgs["decode"]
            self.qk_use_rotate = decode_cfg["qk_config"].get("rotate", True)
            self.av_use_rotate = decode_cfg["av_config"].get("rotate", True)
            self.kv_cache_use_rotate = decode_cfg["kv_cache_config"].get(
                "rotate", True
            )

    def _quantized_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        stage,
        **kwargs,
    ):
        common = dict(
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            qk_bypass=stage["qk_bypass"],
            qk_config=stage["qk_config"],
            av_bypass=stage["av_bypass"],
            av_config=stage["av_config"],
            softmax_bypass=stage["softmax_bypass"],
            softmax_config=stage["softmax_config"],
            **kwargs,
        )
        if self._mx_format == "mxfp":
            function = (
                _eager_attention_forward_mxfp_rotate
                if self._rotate
                else _eager_attention_forward_mxfp
            )
        else:
            function = (
                _eager_attention_forward_mxint_rotate
                if self._rotate
                else _eager_attention_forward_mxint
            )
        if self._rotate:
            common["qk_use_rotate"] = self.qk_use_rotate
            common["av_use_rotate"] = self.av_use_rotate
        return function(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            **common,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ):
        past_key_values = kwargs.pop("past_key_value", past_key_values)
        stage = self._active_stage_cfgs()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        query_states, key_states = _apply_qk_norm(
            self, query_states, key_states, stage
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if stage["rope_bypass"]:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        else:
            query_states, key_states = rope_minifloat(
                query_states,
                key_states,
                cos,
                sin,
                stage["rope_config"],
            )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        def cache_quantizer(key, value, config):
            return kv_cache_mx(
                key,
                value,
                config,
                rotate=self._rotate and self.kv_cache_use_rotate,
            )

        key_states, value_states = self._apply_kv_cache_phase_aware(
            key_states,
            value_states,
            past_key_values,
            cache_kwargs,
            stage,
            cache_quantizer,
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
                sliding_window=self.sliding_window,
                **kwargs,
            )
        else:
            if self.config._attn_implementation != "eager":
                raise ValueError(
                    "quantized Qwen3-MoE attention requires "
                    "_attn_implementation='eager'"
                )
            attn_output, attn_weights = self._quantized_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                stage,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    @classmethod
    def from_self(
        cls, attention: Qwen3MoeAttention, q_config: dict | None = None
    ):
        new = cls(
            config=attention.config,
            layer_idx=attention.layer_idx,
            q_config=q_config,
        )
        parameter = next(attention.parameters())
        new = new.to(device=parameter.device, dtype=parameter.dtype)
        new.load_state_dict(attention.state_dict(), strict=True)
        return new


class Qwen3MoeAttentionMXFP(_PhaseAwareQwen3MoeAttention):
    _mx_format = "mxfp"


class Qwen3MoeAttentionMXInt(_PhaseAwareQwen3MoeAttention):
    _mx_format = "mxint"


class Qwen3MoeAttentionMXFPRotate(Qwen3MoeAttentionMXFP):
    _rotate = True


class Qwen3MoeAttentionMXIntRotate(Qwen3MoeAttentionMXInt):
    _rotate = True
