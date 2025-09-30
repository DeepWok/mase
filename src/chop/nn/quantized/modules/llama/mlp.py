import torch
from torch import nn
import math

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from typing import Optional, Tuple

from transformers.models.llama.modeling_llama import LlamaMLP, ACT2FN

class LlamaMLPLSQInteger(LlamaMLP):
    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config)
        self.config = config
        self.q_config = q_config
        self.layer_idx = layer_idx
        # NOTE: The only change from the original RobertaOutput is the quantization of the dense layer
        # Preserving the original layer architecture for state_dict compatibility
        self.gate_dense_quan = LSQInteger(level=q_config["level"], sym=False)
        self.up_dense_quan = LSQInteger(level=q_config["level"], sym=False)
        self.down_dense_quan = LSQInteger(level=q_config["level"], sym=True)

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        up = self.up_proj(x)
        up = self.up_dense_quan(up)
        gate = self.gate_proj(x)
        gate = self.gate_dense_quan(gate)

        down_proj = self.down_proj(self.act_fn(gate) * up)
        down_proj = self.down_dense_quan(down_proj)

        return down_proj