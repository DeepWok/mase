"""Qwen3-MoE decoder layer with phase-aware residual rounding."""

from __future__ import annotations

from torch import Tensor
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from chop.nn.quantized.functional.vector import VectorRoundingPolicy
from chop.nn.quantized.modules.phase_config import (
    normalize_phase_q_config,
    resolve_module_phase_config,
)
from chop.nn.quantized.modules.phase_context import get_runtime_phase


class Qwen3MoeDecoderLayerMinifloat(Qwen3MoeDecoderLayer):
    """Apply the active FP_SETTING policy at both residual boundaries."""

    def __init__(self, config, layer_idx: int, q_config: dict | None = None):
        super().__init__(config, layer_idx)
        self.q_config = q_config or {}
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self._phase_policies = {
            phase: VectorRoundingPolicy.from_config(
                resolve_module_phase_config(self.phase_q_config, phase)
            )
            for phase in ("prefill", "decode")
        }
        self.bypass = not self._phase_policies["decode"].enabled

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        policy = self._phase_policies[get_runtime_phase()]
        if policy.enabled:
            hidden_states = policy.round(hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = (
            policy.residual_add(residual, hidden_states)
            if policy.enabled
            else residual + hidden_states
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return (
            policy.residual_add(residual, hidden_states)
            if policy.enabled
            else residual + hidden_states
        )

    @classmethod
    def from_self(
        cls, layer: Qwen3MoeDecoderLayer, q_config: dict | None = None
    ):
        new = cls(
            config=layer.self_attn.config,
            layer_idx=layer.self_attn.layer_idx,
            q_config=q_config,
        )
        parameter = next(layer.parameters())
        new = new.to(device=parameter.device, dtype=parameter.dtype)
        new.load_state_dict(layer.state_dict(), strict=True)
        return new
