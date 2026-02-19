from functools import partial

import torch
from torch import Tensor

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the last dimension (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    seq_len = q.size(-2)
    cos = cos[..., :seq_len, :]
    sin = sin[..., :seq_len, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rope_mxfp(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    config: dict = None,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    x_block_size = config["data_in_block_size"]
    x_exp_bits = config["data_in_exponent_width"]
    x_frac_bits = config["data_in_frac_width"]

    x_quantizer = partial(
        mxfp_quantizer,
        block_size=x_block_size,
        element_exp_bits=x_exp_bits,
        element_frac_bits=x_frac_bits,
        block_dim=-1,
    )

    cos = x_quantizer(cos)
    sin = x_quantizer(sin)
    return _apply_rope(q, k, cos, sin, unsqueeze_dim)


def rope_mxint(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    config: dict = None,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    x_block_size = config["data_in_block_size"]
    x_element_bits = config["data_in_width"]

    x_quantizer = partial(
        mxint_quantizer,
        block_size=x_block_size,
        element_bits=x_element_bits,
        block_dim=-1,
    )

    cos = x_quantizer(cos)
    sin = x_quantizer(sin)
    return _apply_rope(q, k, cos, sin, unsqueeze_dim)


def rope_minifloat(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    config: dict = None,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    x_exp_bits = config["data_in_exponent_width"]
    x_frac_bits = config["data_in_frac_width"]

    x_quantizer = partial(
        minifloat_quantizer_sim,
        minifloat_meta=MinifloatMeta(
            exp_bits=x_exp_bits,
            frac_bits=x_frac_bits,
            is_finite=config.get("data_in_is_finite", True),
            round_mode=config.get("data_in_round_mode", "rn"),
        ),
    )

    cos = x_quantizer(cos)
    sin = x_quantizer(sin)
    return _apply_rope(q, k, cos, sin, unsqueeze_dim)
