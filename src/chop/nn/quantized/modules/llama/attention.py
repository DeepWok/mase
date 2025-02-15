from typing import Callable, List, Optional, Tuple, Union

from chop.nn.quantizers.SNN.LSQ import LSQInteger
import torch
from torch import nn

import math

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    ACT2FN,
    LlamaConfig,
    Cache,
    repeat_kv,
    LlamaForCausalLM,
    LlamaDecoderLayer,
)

import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class LlamaAttentionLSQInteger(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, q_config: dict = None):
        tfwriter = SummaryWriter("runs/llamaAttention" + str(layer_idx))
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.query_quan = LSQInteger(
            level=q_config["level"], sym=True, tfwriter=tfwriter
        )
        self.key_quan = LSQInteger(level=q_config["level"], sym=True, tfwriter=tfwriter)
        self.value_quan = LSQInteger(
            level=q_config["level"], sym=True, tfwriter=tfwriter
        )
        self.attn_quan = LSQInteger(
            level=q_config["level"], sym=False, tfwriter=tfwriter
        )
        self.after_attn_quan = LSQInteger(
            level=q_config["level"], sym=False, tfwriter=tfwriter
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = self.query_quan(query_states)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.key_quan(key_states)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.value_quan(value_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # *: matmul_0
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_weights = self.attn_quan(attn_weights)

        # *: matmul_1
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self.after_attn_quan(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
