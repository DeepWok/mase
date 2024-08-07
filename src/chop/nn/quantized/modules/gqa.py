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
        super().__init__(embed_dim, num_heads, num_kv_heads, bias, device, dtype)
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
        config=None,
        out_config=None,
        floor=False,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_kv_heads, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config

        self.q_projection = LinearInteger(
            in_features=embed_dim,
            out_features=embed_dim,
            config=config,
            out_config=out_config,
            bias=bias,
            floor=floor,
        )

        self.k_projection = LinearInteger(
            in_features=embed_dim,
            out_features=self.kv_dim,
            config=config,
            out_config=out_config,
            bias=bias,
            floor=floor,
        )

        self.v_projection = LinearInteger(
            in_features=embed_dim,
            out_features=self.kv_dim,
            config=config,
            out_config=out_config,
            bias=bias,
            floor=floor,
        )

        qk_matmul_q_config = {
            "data_in_width": out_config["data_out_width"],
            "data_in_frac_width": out_config["data_out_frac_width"],
            "weight_width": out_config["data_out_width"],
            "weight_frac_width": out_config["data_out_frac_width"],
        }
        self.qk_matmul_func = partial(
            matmul_integer,
            config=qk_matmul_q_config,
            out_config=out_config,
            floor=floor,
        )

        softermax_config = {
            "width": out_config["data_out_width"],
            "frac_width": out_config["data_out_frac_width"],
        }
        self.softmax_func = partial(
            fixed_softermax,
            q_config=softermax_config,
            out_q_config=softermax_config,
            dim=-1,
        )

        self.v_matmul_func = partial(
            matmul_integer,
            config=config,
            out_config=out_config,
            floor=floor,
        )

        o_projection_q_config = {
            **config,
            "data_in_width": out_config["data_out_width"],
            "data_in_frac_width": out_config["data_out_frac_width"],
        }
        self.o_projection = LinearInteger(
            in_features=embed_dim,
            out_features=embed_dim,
            config=o_projection_q_config,
            out_config=out_config,
            bias=bias,
            floor=floor,
        )
