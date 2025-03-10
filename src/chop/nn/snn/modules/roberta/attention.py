import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import math


from chop.nn.snn.modules.linear import LinearUnfoldBias
from chop.nn.snn.modules.neuron import ST_BIFNode
from chop.nn.snn.modules.softmax import SoftmaxZIPTF


def multi(x1_t, x2_t, x1_sum_t, x2_sum_t):
    """
    SpikeZip-TF multi
    """
    return (
        x1_sum_t @ x2_t.transpose(-2, -1)
        + x1_t @ x2_sum_t.transpose(-2, -1)
        - x1_t @ x2_t.transpose(-2, -1)
    )


def multi1(x1_t, x2_t, x1_sum_t, x2_sum_t):
    """
    SpikeZip-TF multi
    """
    return x1_sum_t @ x2_t + x1_t @ x2_sum_t - x1_t @ x2_t


class RobertaSelfAttentionZIPTF(nn.Module):
    """
    ST-Spike Transformer Self Attention Module
    """

    def __init__(self, config, q_config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.level = q_config["level"]
        self.neuron_type = q_config["neuron_type"]

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = LinearUnfoldBias(
            config.hidden_size,
            self.all_head_size,
            level=q_config["level"],
            neuron_type=q_config["neuron_type"],
        )
        self.query_IF = ST_BIFNode(q_threshold=1.0, level=q_config["level"], sym=True)
        self.key = LinearUnfoldBias(
            config.hidden_size,
            self.all_head_size,
            level=q_config["level"],
            neuron_type=q_config["neuron_type"],
        )
        self.key_IF = ST_BIFNode(q_threshold=1.0, level=q_config["level"], sym=True)
        self.value = LinearUnfoldBias(
            config.hidden_size,
            self.all_head_size,
            level=q_config["level"],
            neuron_type=q_config["neuron_type"],
        )
        self.value_IF = ST_BIFNode(q_threshold=1.0, level=q_config["level"], sym=True)
        self.attn_IF = ST_BIFNode(q_threshold=1.0, level=q_config["level"], sym=False)
        self.after_attn_IF = ST_BIFNode(
            q_threshold=1.0, level=q_config["level"], sym=False
        )
        self.Ssoftmax = SoftmaxZIPTF()

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def reset(self):
        # print("SAttention reset")
        self.query_IF.reset()
        self.key_IF.reset()
        self.value_IF.reset()
        self.attn_IF.reset()
        self.after_attn_IF.reset()
        self.Ssoftmax.reset()
        self.query.reset()
        self.key.reset()
        self.value.reset()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query_IF(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key_IF(self.key(encoder_hidden_states))
            )
            value_layer = self.transpose_for_scores(
                self.value_IF(self.value(encoder_hidden_states))
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key_IF(self.key(hidden_states)))
            value_layer = self.transpose_for_scores(
                self.value_IF(self.value(hidden_states))
            )
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key_IF(self.key(hidden_states)))
            value_layer = self.transpose_for_scores(
                self.value_IF(self.value(hidden_states))
            )

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = multi(
            query_layer,
            key_layer,
            self.transpose_for_scores(self.query_IF.acc_q * self.query_IF.q_threshold),
            self.transpose_for_scores(self.key_IF.acc_q * self.key_IF.q_threshold),
        )

        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.Ssoftmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        attention_probs = self.attn_IF(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = multi1(
            attention_probs,
            value_layer,
            (self.attn_IF.acc_q * self.attn_IF.q_threshold),
            self.transpose_for_scores(self.value_IF.acc_q * self.value_IF.q_threshold),
        )

        context_layer = self.after_attn_IF(context_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
