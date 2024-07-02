import torch
from torch import Tensor
import torch.nn as nn
import math

from typing import Optional, Tuple
from functools import partial

from chop.nn.quantized.functional.matmul import (
    generic_matmul_integer,
)
from chop.nn.quantizers.integer import integer_quantizer


class _BertSelfAttentionHeadBase(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # ! TO DO: replace these with quantized functions?
        self.matmul = torch.matmul
        self.softmax = nn.functional.softmax

    def self_attention_head(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tensor:
        attention_scores = self.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.matmul(attention_probs, value_layer)
        return context_layer

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tensor:
        return self.self_attention_head(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask=attention_mask,
        )


class BertSelfAttentionHeadInteger(_BertSelfAttentionHeadBase):
    def __init__(self, config, q_config: dict = None) -> None:
        super().__init__(config)

        self.query_quantizer = partial(
            integer_quantizer,
            **q_config,
        )
        self.key_quantizer = partial(integer_quantizer, **q_config)
        self.value_quantizer = partial(integer_quantizer, **q_config)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tensor:
        query_layer = self.query_quantizer(query_layer)
        key_layer = self.key_quantizer(key_layer)
        value_layer = self.value_quantizer(value_layer)

        return self.self_attention_head(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask=attention_mask,
        )
