import torch
from torch import nn

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantized.functional.vector import VectorRoundingPolicy
from chop.nn.quantized.modules.phase_context import get_runtime_phase
from chop.nn.quantized.modules.phase_config import (
    normalize_phase_q_config,
    resolve_module_phase_config,
)

from transformers.models.llama.modeling_llama import LlamaRMSNorm


class LlamaRMSNormLSQInteger(LlamaRMSNorm):
    def __init__(self, config=None, layer_idx=None, q_config: dict = None):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps
        self.quant_after_ln = LSQInteger(level=q_config["level"], sym=True)
        self.layer_idx = layer_idx

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.quant_after_ln(self.weight * hidden_states.to(input_dtype))

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def build_vector_phase_policies(
    cfg: dict,
) -> tuple[VectorRoundingPolicy, VectorRoundingPolicy]:
    """Build weight and activation policies for one RMSNorm phase."""

    if cfg.get("bypass", False):
        disabled = VectorRoundingPolicy.disabled()
        return disabled, disabled

    x_policy = (
        VectorRoundingPolicy.disabled()
        if cfg.get("data_in_bypass", False)
        else VectorRoundingPolicy.from_config(cfg)
    )
    if cfg.get("weight_bypass", False):
        weight_policy = VectorRoundingPolicy.disabled()
    elif "weight_exponent_width" in cfg and "weight_frac_width" in cfg:
        weight_policy = VectorRoundingPolicy.from_config(
            {
                "data_in_exponent_width": cfg["weight_exponent_width"],
                "data_in_frac_width": cfg["weight_frac_width"],
                "data_in_is_finite": cfg.get("weight_is_finite", False),
                "data_in_round_mode": cfg.get("weight_round_mode", "rn"),
            }
        )
    else:
        weight_policy = x_policy
    return weight_policy, x_policy


class LlamaRMSNormMinifloat(LlamaRMSNorm):
    """Minifloat-quantized LlamaRMSNorm with per-phase quantizers.

    Weight and input use minifloat quantization at forward time; quantizers
    are built once per phase at construction (not per forward call, which
    matters during token-by-token decoding).
    """

    def __init__(self, config=None, layer_idx=None, q_config: dict = None):
        super().__init__(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.variance_epsilon = config.rms_norm_eps
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self._phase_policies = {
            phase: build_vector_phase_policies(
                resolve_module_phase_config(self.phase_q_config, phase)
            )
            for phase in ("prefill", "decode")
        }
        # Legacy attr mirrors the decode side (the quantised chip).
        self.bypass = resolve_module_phase_config(self.phase_q_config, "decode").get(
            "bypass", False
        )

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
