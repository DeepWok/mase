import torch
from torch import nn
import math

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from typing import Optional, Tuple


from transformers.models.roberta.modeling_roberta import RobertaIntermediate


class RobertaIntermediateLSQInteger(RobertaIntermediate):
    def __init__(self, config, q_config: dict = None):
        super().__init__(config)
        self.config = config
        self.q_config = q_config
        # NOTE: The only change from the original RobertaOutput is the quantization of the dense layer
        # Preserving the original layer architecture for state_dict compatibility
        self.dense_quan = LSQInteger(level=q_config["level"], sym=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dense_quan(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
