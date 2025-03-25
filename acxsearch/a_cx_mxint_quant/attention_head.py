import torch
from torch import Tensor
import torch.nn as nn
import math

from typing import Optional, Tuple
from functools import partial

from chop.nn.quantized.functional.matmul import (
    generic_matmul_integer,
)
from chop.nn.quantizers.integer import integer_quantizer, integer_floor_quantizer
from .quantizers import mxint_quant_block

class _ViTSelfAttentionHeadBase(torch.nn.Module):
    def __init__(self, dim, num_heads, attn_drop) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attn_drop)

        self.matmul1 = torch.matmul
        self.matmul2 = torch.matmul
        self.mult_data = torch.tensor(1 / math.sqrt(dim))
        self.act = nn.functional.softmax

    def self_attention_head(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> Tensor:
        attention_scores = self.matmul1(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.mult_data

        # Normalize the attention scores to probabilities.
        attention_probs = self.act(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = self.matmul2(attention_probs, value_layer)
        return context_layer

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> Tensor:
        return self.self_attention_head(
            query_layer=query_layer, key_layer=key_layer, value_layer=value_layer
        )

from .linear import MXIntLinear, fast_linear
from .quantizers import mxint_hardware

class MXIntMatMul(nn.Module):
    def __init__(self, q_config=None):
        super().__init__()
        assert q_config is not None, "q_config cannot be None"
        self.q_config = q_config
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        qx, _, _ = mxint_hardware(
            x, 
            q_config = {
                "width": self.q_config["data_in_width"],
                "exponent_width": self.q_config["data_in_exponent_width"],
            },
            parallelism = self.q_config["data_in_parallelism"]
        )
        qy, _, _ = mxint_hardware(
            y,
            q_config = {
                "width": self.q_config["weight_width"],
                "exponent_width": self.q_config["weight_exponent_width"],
            },
            parallelism = self.q_config["weight_parallelism"]
        )

        out = qx @ qy
        out, _, _ = mxint_hardware(
            out,
            q_config = {
                "width": self.q_config["data_out_width"],
                "exponent_width": self.q_config["data_out_exponent_width"],
            },
            parallelism = self.q_config["data_out_parallelism"]
        )
        return out

from .softmax import MXIntSoftmax, IntSoftmax

class MXIntViTAttentionHead(_ViTSelfAttentionHeadBase):
    def __init__(
        self, dim, num_heads, attn_drop=0.0, q_config: dict = None, floor=False
    ) -> None:
        super().__init__(dim, num_heads, attn_drop)
        self.dropout = nn.Dropout(attn_drop)


        self.matmul1 = torch.matmul
        self.matmul2 = torch.matmul
        self.act = MXIntSoftmax(q_config=q_config)
        self.mult_data = torch.tensor(1 / math.sqrt(dim))

class IntViTAttentionHead(_ViTSelfAttentionHeadBase):
    def __init__(
        self, dim, num_heads, attn_drop=0.0, q_config: dict = None, floor=False
    ) -> None:
        super().__init__(dim, num_heads, attn_drop)
        self.dropout = nn.Dropout(attn_drop)


        self.matmul1 = torch.matmul
        self.matmul2 = torch.matmul
        self.act = IntSoftmax(q_config=q_config)
        self.mult_data = torch.tensor(1 / math.sqrt(dim))
# class ViTSelfAttentionHeadInteger(_ViTSelfAttentionHeadBase):
#     def __init__(
#         self, dim, num_heads, attn_drop=0.0, q_config: dict = None, floor=False
#     ) -> None:
#         super().__init__(dim, num_heads, attn_drop)
#         base_quantizer = integer_floor_quantizer if floor else integer_quantizer
#         self.query_quantizer = partial(
#             base_quantizer,
#             width=q_config["query_width"],
#             frac_width=q_config["query_frac_width"],
#         )
#         self.key_quantizer = partial(
#             base_quantizer,
#             width=q_config["key_width"],
#             frac_width=q_config["key_frac_width"],
#         )
#         self.value_quantizer = partial(
#             base_quantizer,
#             width=q_config["value_width"],
#             frac_width=q_config["value_frac_width"],
#         )
#         self.matmul1 = partial(
#             generic_matmul_integer,
#             config={
#                 "data_in_width": q_config["query_width"],
#                 "data_in_frac_width": q_config["query_frac_width"],
#                 "weight_width": q_config["key_width"],
#                 "weight_frac_width": q_config["key_frac_width"],
#             },
#             out_config={
#                 "data_out_width": q_config["qkmm_out_width"],
#                 "data_out_frac_width": q_config["qkmm_out_frac_width"],
#             },
#             floor=floor,
#         )
#         self.act = partial(
#             softmax_integer,
#             config={
#                 "data_in_width": q_config["qkmm_out_width"],
#                 "data_in_frac_width": q_config["qkmm_out_frac_width"],
#                 "data_in_exp_width": q_config["softmax_exp_width"],
#                 "data_in_exp_frac_width": q_config["softmax_exp_frac_width"],
#                 "data_out_frac_width": q_config["softmax_out_frac_width"],
#                 "mult_data": self.mult_data,
#             },
#             floor=floor,
#         )
#         self.mult_data = torch.tensor(1)
#         self.matmul2 = partial(
#             generic_matmul_integer,
#             config={
#                 "data_in_width": q_config["softmax_out_frac_width"] + 2,
#                 "data_in_frac_width": q_config["softmax_out_frac_width"],
#                 "weight_width": q_config["value_width"],
#                 "weight_frac_width": q_config["value_frac_width"],
#             },
#             out_config={
#                 "data_out_width": q_config["svmm_out_width"],
#                 "data_out_frac_width": q_config["svmm_out_frac_width"],
#             },
#             floor=floor,
#         )
