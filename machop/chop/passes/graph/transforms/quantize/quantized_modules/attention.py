from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F
from .utils import get_stats, quantiser_passthrough

from transformers.models.bert.modeling_bert import BertAttention

from .linear import LinearInteger
from .layer_norm import LayerNormInteger

from typing import Optional, Tuple


class _BertSelfAttentionBase(torch.nn.Module):
    def __init__(
        self, config, q_config: dict = None, position_embedding_type=None
    ) -> None:
        super().__init__()
        self.bypass = False

        self.q_config = q_config
        self.attention = BertAttention(config)

        self.hidden_states_quantizer = None
        self.weight_query_quantizer = None
        self.weight_key_quantizer = None
        self.weight_value_quantizer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tensor:
        return self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )


class BertSelfAttentionInteger(_BertSelfAttentionBase):
    def __init__(
        self, config, q_config: dict = None, position_embedding_type=None
    ) -> None:
        super().__init__(config, q_config, position_embedding_type)
        self.attention.self.query = LinearInteger(
            config.hidden_size,
            config.all_head_size,
            config=self.q_config,
        )
        self.attention.self.key = LinearInteger(
            config.hidden_size,
            config.all_head_size,
            config=self.q_config,
        )
        self.attention.self.value = LinearInteger(
            config.hidden_size,
            config.all_head_size,
            config=self.q_config,
        )
        self.attention.output.dense = LinearInteger(
            config.hidden_size,
            config.all_head_size,
            config=self.q_config,
        )
        self.attention.output.LayerNorm = LayerNormInteger(
            config.hidden_size,
            eps=config.layer_norm_eps,
            config=self.q_config,
        )
