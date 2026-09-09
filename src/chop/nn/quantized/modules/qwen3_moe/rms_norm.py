"""Phase-aware vector rounding for Qwen3-MoE RMSNorm."""

from chop.nn.quantized.modules.llama.rms_norm import build_vector_phase_policies
from chop.nn.quantized.modules.phase_config import (
    normalize_phase_q_config,
    resolve_module_phase_config,
)
from chop.nn.quantized.modules.phase_context import get_runtime_phase
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm


class Qwen3MoeRMSNormMinifloat(Qwen3MoeRMSNorm):
    def __init__(
        self,
        config=None,
        layer_idx=None,
        q_config: dict | None = None,
        *,
        hidden_size: int | None = None,
        eps: float | None = None,
    ):
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        eps = eps if eps is not None else config.rms_norm_eps
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self._phase_policies = {
            phase: build_vector_phase_policies(
                resolve_module_phase_config(self.phase_q_config, phase)
            )
            for phase in ("prefill", "decode")
        }
        self.bypass = resolve_module_phase_config(
            self.phase_q_config, "decode"
        ).get("bypass", False)

    def forward(self, hidden_states):
        weight_policy, input_policy = self._phase_policies[get_runtime_phase()]
        if not weight_policy.enabled and not input_policy.enabled:
            return super().forward(hidden_states)
        return input_policy.rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            weight_policy=weight_policy,
        )

    @classmethod
    def from_self(cls, norm: Qwen3MoeRMSNorm, q_config: dict | None = None):
        new = cls(
            q_config=q_config,
            hidden_size=norm.weight.numel(),
            eps=norm.variance_epsilon,
        )
        new = new.to(device=norm.weight.device, dtype=norm.weight.dtype)
        new.load_state_dict(norm.state_dict(), strict=True)
        return new
