import torch
from torch import nn
import math

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from typing import Optional, Tuple

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