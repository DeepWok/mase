"""Llama RMSNorm quantization with phase-aware dispatch.

Decode is intentionally forced to FP in step-1 for deterministic integration.
Runtime phase is supplied by decoder-layer pre-hooks via phase context.
"""

from functools import partial

import torch
from torch import Tensor, nn

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim
from chop.nn.quantized.modules.phase_context import get_active_phase
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
        if self.decode_policy != "fp_only":
            raise ValueError(
                "Step-1 integration only supports decode_policy='fp_only' "
                f"for {self.__class__.__name__}, got {self.decode_policy!r}."
            )

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
        phase = get_active_phase()
        sub_cfg, decode_policy = get_phase_subconfig(self.phase_q_config, phase)
        bypass = sub_cfg.get("bypass", False)
        if phase == "decode" and decode_policy == "fp_only":
            # Why force bypass here:
            # step-1 intentionally keeps decode fully FP for stability and
            # backward compatibility while still accepting phase-shaped configs.
            bypass = True

        w_quantizer = self._build_weight_quantizer(sub_cfg, bypass)
        x_quantizer = self._build_input_quantizer(sub_cfg, bypass)

        input_dtype = hidden_states.dtype
        if x_quantizer is not None:
            hidden_states = x_quantizer(hidden_states)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = (
            w_quantizer(self.weight)
            if w_quantizer is not None
            else self.weight
        )
        return weight * hidden_states.to(input_dtype)
