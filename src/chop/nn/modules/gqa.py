import math
from functools import partial

import torch
from torch import nn, Tensor

from chop.nn.functional.softermax import softermax


# Copied from transformers.models.llama.modeling_llama
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) based on this paper:
    https://arxiv.org/abs/2305.13245v3

    Using similar API to torch.nn.MultiheadAttention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        # Assign params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.bias = bias
        self.head_dim = self.embed_dim // self.num_heads
        self.group_size = self.num_heads // self.num_kv_heads
        self.kv_dim = self.embed_dim // self.group_size

        self._assert_gqa_config()

        # Q is projected to embedding dim
        self.q_projection = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )

        # K & V are only projected to (embed_dim / group_size) and their
        # outputs are then duplicated group_size times
        self.k_projection = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.kv_dim,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )
        self.v_projection = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.kv_dim,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )

        # Output projection
        self.o_projection = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=self.bias,
            device=device,
            dtype=dtype,
        )

        # Explicitly define functions so quantized version of GQA can override
        self.softmax_func = partial(softermax, dim=-1)
        self.qk_matmul_func = torch.matmul
        self.v_matmul_func = torch.matmul

    def _assert_gqa_config(self):
        assert (
            self.group_size * self.num_kv_heads == self.num_heads
        ), "Number of heads must be divisible by number of KV heads!"
        assert (
            self.group_size * self.kv_dim == self.embed_dim
        ), "Embedding dimension must be divisible by number of KV heads!"
        assert (
            self.num_heads * self.head_dim == self.embed_dim
        ), "Embedding dimension must be divisible by number of heads!"

    def _qkv_states(self, x: Tensor, batch_size: int, seq_len: int):
        query = (
            self.q_projection(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_projection(x)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_projection(x)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        return query, key, value

    # def _rotary_embeddings(self, x: Tensor):
    #     # TODO: Implement RoPE
    #     return x

    def _attention_mechanism(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_size: int,
        seq_len: int,
    ):
        key = repeat_kv(key, n_rep=self.group_size)
        value = repeat_kv(value, n_rep=self.group_size)

        attn_weights = self.qk_matmul_func(
            query, key.transpose(2, 3) / math.sqrt(self.head_dim)
        )
        attn_weights = self.softmax_func(attn_weights)
        attn_output = self.v_matmul_func(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        out = self.o_projection(attn_output)
        return out

    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.shape

        query, key, value = self._qkv_states(x, batch_size, seq_len)

        # TODO: Missing rotary embeddings to Llama

        out = self._attention_mechanism(query, key, value, batch_size, seq_len)

        return out


if __name__ == "__main__":
    BATCH = 10
    SEQ_LEN = 16
    EMBED_DIM = 256
    NUM_HEADS = 8
    GROUPS = 4

    gqa_module = GroupedQueryAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=GROUPS,
    )

    x_in = torch.rand(BATCH, SEQ_LEN, EMBED_DIM)
    y_out = gqa_module(x_in)
    print(y_out)
