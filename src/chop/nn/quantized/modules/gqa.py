from functools import partial

import torch
from chop.nn.modules import GroupedQueryAttention
from chop.nn.quantized.modules import LinearInteger
from chop.nn.quantized.functional import fixed_softermax, matmul_integer


class _GroupedQueryAttentionBase(GroupedQueryAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, num_kv_heads, bias, device, dtype
        )
        self.bypass = False


class GroupedQueryAttentionInteger(_GroupedQueryAttentionBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        device=None,
        dtype=None,
        # Quantization Configs
        linear_q_config: dict = None,
        linear_out_q_config: dict = None,
        softermax_out_q_config: dict = None,
        qk_matmul_out_q_config: dict = None,
        v_matmul_out_q_config: dict = None,
        # Rounding Mode
        floor=False,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, num_kv_heads, bias, device, dtype
        )

        self.q_projection = LinearInteger(
            in_features=embed_dim,
            out_features=embed_dim,
            config=linear_q_config,
            out_config=linear_out_q_config,
            bias=bias,
            floor=floor,
        )

        self.k_projection = LinearInteger(
            in_features=embed_dim,
            out_features=self.kv_dim,
            config=linear_q_config,
            out_config=linear_out_q_config,
            bias=bias,
            floor=floor,
        )

        self.v_projection = LinearInteger(
            in_features=embed_dim,
            out_features=self.kv_dim,
            config=linear_q_config,
            out_config=linear_out_q_config,
            bias=bias,
            floor=floor,
        )

        qk_matmul_q_config = {
            "data_in_width": linear_out_q_config["data_out_width"],
            "data_in_frac_width": linear_out_q_config["data_out_frac_width"],
            "weight_width": linear_out_q_config["data_out_width"],
            "weight_frac_width": linear_out_q_config["data_out_frac_width"],
        }
        self.qk_matmul_func = partial(
            matmul_integer,
            config=qk_matmul_q_config,
            out_config=qk_matmul_out_q_config,
            floor=floor,
        )

        softermax_q_config = {
            "width": qk_matmul_out_q_config["data_out_width"],
            "frac_width": qk_matmul_out_q_config["data_out_frac_width"],
        }
        self.softmax_func = partial(
            fixed_softermax,
            q_config=softermax_q_config,
            out_q_config=softermax_out_q_config,
            dim=-1,
        )

        v_matmul_q_config = {
            "data_in_width": softermax_out_q_config["width"],
            "data_in_frac_width": softermax_out_q_config["frac_width"],
            "weight_width": linear_out_q_config["data_out_width"],
            "weight_frac_width": linear_out_q_config["data_out_frac_width"],
        }
        self.v_matmul_func = partial(
            matmul_integer,
            config=v_matmul_q_config,
            out_config=v_matmul_out_q_config,
            floor=floor,
        )

        o_projection_q_config = {
            **linear_q_config,
            "data_in_width": linear_out_q_config["data_out_width"],
            "data_in_frac_width": linear_out_q_config["data_out_frac_width"],
        }
        self.o_projection = LinearInteger(
            in_features=embed_dim,
            out_features=embed_dim,
            config=o_projection_q_config,
            out_config=linear_out_q_config,
            bias=bias,
            floor=floor,
        )
