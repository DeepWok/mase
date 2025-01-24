import torch
from torch import nn
import math

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from typing import Optional, Tuple


from transformers.models.roberta.modeling_roberta import RobertaClassificationHead


class RobertaClassificationHeadLSQInteger(RobertaClassificationHead):
    def __init__(self, config, q_config: dict = None):
        super().__init__(config)
        self.config = config
        self.q_config = q_config
        # NOTE: The only change from the original RobertaOutput is the quantization of the dense layer
        # Preserving the original layer architecture for state_dict compatibility
        self.dense_quan = LSQInteger(level=q_config["level"], sym=False)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dense_quan(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
