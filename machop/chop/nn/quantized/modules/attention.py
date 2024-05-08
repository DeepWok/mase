from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from transformers.models.bert.modeling_bert import BertSelfAttention

from chop.passes.graph.transforms.quantize.quantized_modules.linear import (
    LinearInteger,
)
from chop.nn.quantized import fixed_softermax
from chop.passes.graph.transforms.quantize.quantized_funcs import matmul_integer

from typing import Optional, Tuple


class _BertSelfAttentionBase(BertSelfAttention):
    def __init__(
        self,
        config,
        q_config: dict = None,
        out_q_config: dict = None,
        position_embedding_type=None,
        bias=True,
    ) -> None:
        super().__init__(config, position_embedding_type)
        self.bypass = False
        self.q_config = q_config
        self.out_q_config = out_q_config
        self.bias = bias


class BertSelfAttentionInteger(_BertSelfAttentionBase):
    def __init__(
        self,
        config,
        q_config: dict = None,
        out_q_config: dict = None,
        position_embedding_type=None,
        bias=True,
        floor=False,
    ) -> None:
        super().__init__(
            config, q_config, out_q_config, position_embedding_type, bias=bias
        )
        self.query = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        self.key = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        self.value = LinearInteger(
            config.hidden_size,
            config.hidden_size,
            config=q_config,
            out_config=out_q_config,
            bias=bias,
            floor=floor,
        )
        # * Matmul is used for Q @ K^T and Scores @ V where the input values have already
        # * been casted to the output precision, so we provide the output precision to the
        # * software model
        self.matmul = partial(
            matmul_integer,
            config={
                "data_in_width": self.out_q_config["data_out_width"],
                "data_in_frac_width": self.out_q_config["data_out_frac_width"],
                "weight_width": self.out_q_config["data_out_width"],
                "weight_frac_width": self.out_q_config["data_out_frac_width"],
            },
            out_config={
                "data_out_width": self.out_q_config["data_out_width"],
                "data_out_frac_width": self.out_q_config["data_out_frac_width"],
            },
            floor=floor,
        )
