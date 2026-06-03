"""Llama RMSNorm quantization with phase-aware dispatch.

Default decode behavior remains FP-only for backward compatibility.
`quantized` decode is opt-in via phase config and policy.
"""

from functools import partial

import torch
from torch import Tensor, nn

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim
from chop.nn.quantized.modules.phase_context import get_runtime_phase
from chop.nn.quantized.modules.phase_config import (
    get_phase_subconfig,
    normalize_phase_q_config,
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


class LlamaRMSNormMinifloat(LlamaRMSNorm):
    """Minifloat-quantized LlamaRMSNorm. Weight and input use minifloat quantization at forward time."""

    def __init__(self, config=None, layer_idx=None, q_config: dict = None):
        super().__init__(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        self.variance_epsilon = config.rms_norm_eps
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]

    @staticmethod
    def _build_weight_quantizer(sub_cfg: dict, bypass: bool):
        """Build weight quantizer for current phase config."""

        if bypass or sub_cfg.get("weight_bypass", False):
            return None
        return partial(
            minifloat_quantizer_sim,
            minifloat_meta=MinifloatMeta(
                exp_bits=sub_cfg["weight_exponent_width"],
                frac_bits=sub_cfg["weight_frac_width"],
                is_finite=sub_cfg.get("weight_is_finite", True),
                round_mode=sub_cfg.get("weight_round_mode", "rn"),
            ),
        )

    @staticmethod
    def _build_input_quantizer(sub_cfg: dict, bypass: bool):
        """Build input quantizer for current phase config."""

        if bypass or sub_cfg.get("data_in_bypass", False):
            return None
        return partial(
            minifloat_quantizer_sim,
            minifloat_meta=MinifloatMeta(
                exp_bits=sub_cfg["data_in_exponent_width"],
                frac_bits=sub_cfg["data_in_frac_width"],
                is_finite=sub_cfg.get("data_in_is_finite", True),
                round_mode=sub_cfg.get("data_in_round_mode", "rn"),
            ),
        )

    def forward(self, hidden_states):
        runtime_phase = get_runtime_phase()
        phase_subconfig, decode_policy = get_phase_subconfig(
            self.phase_q_config, runtime_phase
        )
        bypass = phase_subconfig.get("bypass", False)
        if runtime_phase == "decode" and decode_policy == "fp_only":
            bypass = True

        w_quantizer = self._build_weight_quantizer(phase_subconfig, bypass)
        x_quantizer = self._build_input_quantizer(phase_subconfig, bypass)

        input_dtype = hidden_states.dtype
        if x_quantizer is not None:
            hidden_states = x_quantizer(hidden_states)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = w_quantizer(self.weight) if w_quantizer is not None else self.weight
        return weight * hidden_states.to(input_dtype)
