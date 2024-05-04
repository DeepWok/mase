from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from transformers.models.bert.modeling_bert import BertSelfAttention

from chop.passes.graph.transforms.quantize.quantized_modules.linear import (
    LinearInteger,
)

from typing import Optional, Tuple


class _BertSelfAttentionBase(BertSelfAttention):
    def __init__(
        self, config, q_config: dict = None, position_embedding_type=None
    ) -> None:
        super().__init__(config, position_embedding_type)
        self.bypass = False
        self.q_config = q_config


class BertSelfAttentionInteger(_BertSelfAttentionBase):
    def __init__(
        self, config, q_config: dict = None, position_embedding_type=None
    ) -> None:
        super().__init__(config, q_config, position_embedding_type)
        self.query = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=self.q_config,
        )
        self.key = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=self.q_config,
        )
        self.value = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=self.q_config,
        )
