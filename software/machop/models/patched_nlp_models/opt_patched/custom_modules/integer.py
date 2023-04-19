import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .....modify.quantizers.functions import bmm_integer
from .....modify.quantizers.layers import LinearInteger
from .....utils import copy_weights
from ..modeling_opt_patched import OPTAttentionPatched


class OPTAttentionInteger(nn.Module):
    """
    - FX-traceable Multi-headed attention from 'Attention Is All You Need' paper
    - This module includes multi-head (k, q, v linear, attention), concat, and attention output linear
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        config: Dict = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = self._construct_essential_config(config)

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = LinearInteger(
            embed_dim, embed_dim, bias=bias, config=self.config["k_proj"]
        )
        self.v_proj = LinearInteger(
            embed_dim, embed_dim, bias=bias, config=self.config["v_proj"]
        )
        self.q_proj = LinearInteger(
            embed_dim, embed_dim, bias=bias, config=self.config["q_proj"]
        )
        self.out_proj = LinearInteger(
            embed_dim, embed_dim, bias=bias, config=self.config["out_proj"]
        )

    def _construct_essential_config(self, config):
        assert "default" in config
        if "k_proj" not in config:
            config["k_proj"] = copy.deepcopy(config["default"])
        if "v_proj" not in config:
            config["v_proj"] = copy.deepcopy(config["default"])
        if "q_proj" not in config:
            config["q_proj"] = copy.deepcopy(config["default"])
        if "out_proj" not in config:
            config["out_proj"] = copy.deepcopy(config["default"])
        if "bmm_attention" not in config:
            config["bmm_attention"] = copy.deepcopy(config["default"])
        if "bmm_context" not in config:
            config["bmm_context"] = copy.deepcopy(config["default"])
        return config

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """
        [B, N, d_h] -> [B, h, N, d_k]
        B: batch size
        N: sequence len
        d_h: model hidden size/ embed size
        d_k: hidden size per head/ head size, d_k = d_h / h
        h: the number of heads
        """
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.shape

        # Q = X @ W_Q^T * (1/sqrt(k_d))
        query_states = self.q_proj(hidden_states) * self.scaling
        # K = X @ W_K^T
        # [B, N, d_h] -> [B, h, N, d_k]
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        # V = X @ W_V^T
        # [B, N, d_h] -> [B, h, N, d_k]
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # Q: [B, N, d_h] -> [B, h, N, d_k] -> [B*h, N, d_k]
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # K and V: [B, h, N, d_k] -> [B*h, N, d_k]
        # this view is to use bmm, which only allows 3D input where the 1st dim is batch size
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]

        # bmm 0
        # compute attention: A = Q @ K^T
        # attn_weights: [B*h, N, N]
        attn_weights = bmm_integer(
            query_states,
            key_states.transpose(1, 2),
            config=self.config["bmm_attention"],
        )

        assert attention_mask is not None
        assert attention_mask.size() == (bsz, 1, tgt_len, src_len)

        # attn_weights: [B*h, N, N] -> [B, h, N, N]
        # attention_mask: the masked attn_weights is close to -inf. This -inf comes from attention_mask
        attn_weights = (
            attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        )
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        # attn_weights: [B, h, N, N] -> [B*h, N, N]
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        assert layer_head_mask is None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # bmm context
        # apply attention to V to get context: Context = A @ V
        # attn_output (context): [B*h, N, d_k]
        attn_output = bmm_integer(
            attn_probs, value_states, config=self.config["bmm_context"]
        )
        # attn_output: -> [B, h, N, d_k] -> [B, N, h, d_k]
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # attn_output: -> [B, N, d_h]
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        # output project: [B, N, d_h] -> [B, N, d_h]
        attn_output = self.out_proj(attn_output)

        attn_weights_reshaped = None
        return attn_output, attn_weights_reshaped
