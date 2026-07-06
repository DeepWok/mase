from functools import partial

import torch
from torch import Tensor, nn

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim
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


def build_minifloat_phase_quantizers(cfg: dict):
    """Return (w_quantizer, x_quantizer) for one phase sub-config.

    Shared by the Llama and Qwen3 minifloat RMSNorm modules so the config-key
    handling cannot drift between architectures.
    """
    bypass = cfg.get("bypass", False)

    if not bypass and not cfg.get("weight_bypass", False):
        w_quantizer = partial(
            minifloat_quantizer_sim,
            minifloat_meta=MinifloatMeta(
                exp_bits=cfg["weight_exponent_width"],
                frac_bits=cfg["weight_frac_width"],
                is_finite=cfg.get("weight_is_finite", True),
                round_mode=cfg.get("weight_round_mode", "rn"),
            ),
        )
    else:
        w_quantizer = None

    if not bypass and not cfg.get("data_in_bypass", False):
        x_quantizer = partial(
            minifloat_quantizer_sim,
            minifloat_meta=MinifloatMeta(
                exp_bits=cfg["data_in_exponent_width"],
                frac_bits=cfg["data_in_frac_width"],
                is_finite=cfg.get("data_in_is_finite", True),
                round_mode=cfg.get("data_in_round_mode", "rn"),
            ),
        )
    else:
        x_quantizer = None

    return w_quantizer, x_quantizer


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
        self._phase_quantizers = {
            phase: build_minifloat_phase_quantizers(
                resolve_module_phase_config(self.phase_q_config, phase)
            )
            for phase in ("prefill", "decode")
        }
        # Legacy attr mirrors the decode side (the quantised chip).
        self.bypass = resolve_module_phase_config(self.phase_q_config, "decode").get(
            "bypass", False
        )

    def forward(self, hidden_states):
        w_quantizer, x_quantizer = self._phase_quantizers[get_runtime_phase()]
        input_dtype = hidden_states.dtype
        if x_quantizer is not None:
            hidden_states = x_quantizer(hidden_states)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = w_quantizer(self.weight) if w_quantizer is not None else self.weight
        return weight * hidden_states.to(input_dtype)
