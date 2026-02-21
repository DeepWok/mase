from functools import partial

import torch
from torch import Tensor, nn

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim

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
        self.q_config = q_config or {}
        self.variance_epsilon = config.rms_norm_eps
        self.bypass = self.q_config.get("bypass", False)
        self.weight_bypass = self.q_config.get("weight_bypass", False)
        self.data_in_bypass = self.q_config.get("data_in_bypass", False)

        if not self.bypass and not self.weight_bypass:
            self.w_quantizer = partial(
                minifloat_quantizer_sim,
                minifloat_meta=MinifloatMeta(
                    exp_bits=self.q_config["weight_exponent_width"],
                    frac_bits=self.q_config["weight_frac_width"],
                    is_finite=self.q_config.get("weight_is_finite", True),
                    round_mode=self.q_config.get("weight_round_mode", "rn"),
                ),
            )
        else:
            self.w_quantizer = None

        if not self.bypass and not self.data_in_bypass:
            self.x_quantizer = partial(
                minifloat_quantizer_sim,
                minifloat_meta=MinifloatMeta(
                    exp_bits=self.q_config["data_in_exponent_width"],
                    frac_bits=self.q_config["data_in_frac_width"],
                    is_finite=self.q_config.get("data_in_is_finite", True),
                    round_mode=self.q_config.get("data_in_round_mode", "rn"),
                ),
            )
        else:
            self.x_quantizer = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        if self.x_quantizer is not None:
            hidden_states = self.x_quantizer(hidden_states)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.w_quantizer(self.weight) if self.w_quantizer is not None else self.weight
        return weight * hidden_states.to(input_dtype)
