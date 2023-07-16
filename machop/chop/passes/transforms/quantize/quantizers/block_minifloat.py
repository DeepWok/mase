from typing import List

import torch
from torch import Tensor

from .utils import block, my_clamp, unblock
from .minifloat import minifloat_ieee_quantizer


def _block_minifloat_quantize(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias_width: int,
    block_size: List[int] = 16,
    skip_first_dim: bool = False,
):
    """
    - Convert IEEE FP32/64 to Block Minifloat (BM), where an exponent bias is shared over all elements in a block
    - `2**-bias_shared x [(-1)^s1 x 2^exponent1 x mantissa1, (-1)^s2 x 2^exponent2 x mantissa2, ...]`
    - See https://openreview.net/forum?id=6zaTwpNSsQ2

    ---
    - forward: convert IEEE FP32/64 to BM
    - backward: STE

    ---
    - `width`: the number of bits (1 sign bit + exponent_bits + mantissa_bits)
    - `exponent_width`: the number of exponent_bits
    - `exponent_bias_width`: the number of bits of the shared exponent bias
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )

    # fill zeros to avoid log2(0) = -inf
    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    per_block_exponent_bias = my_clamp(
        torch.floor(torch.log2(per_block_max)), 0, 2**exponent_bias_width - 1
    )
    per_block_bm_x = minifloat_ieee_quantizer(
        blocked_x,
        width=width,
        exponent_width=exponent_width,
        exponent_bias=per_block_exponent_bias,
    )

    bm_x = unblock(
        per_block_bm_x,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )
    return bm_x


class BlockMinifloatQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        width: int,
        exponent_width: int,
        exponent_bias_width: int,
        block_size: List[int] = 16,
        skip_first_dim: bool = False,
    ):
        return _block_minifloat_quantize(
            x,
            width=width,
            exponent_width=exponent_width,
            exponent_bias_width=exponent_bias_width,
            block_size=block_size,
            skip_first_dim=skip_first_dim,
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor,
        width: int,
        exponent_width: int,
        exponent_bias_width: int,
        block_size: List[int] = 16,
        skip_first_dim: bool = False,
    ):
        return grad_output, None, None, None, None, None


def block_minifloat_quantizer(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias_width: int,
    block_size: List[int] = 16,
    skip_first_dim: bool = False,
):
    """
    - Convert IEEE FP32/64 to Block Minifloat (BM), where an exponent bias is shared over all elements in a block
    - `2**-bias_shared x [(-1)^s1 x 2^exponent1 x mantissa1, (-1)^s2 x 2^exponent2 x mantissa2, ...]`
    - See https://openreview.net/forum?id=6zaTwpNSsQ2

    ---
    - forward: convert IEEE FP32/64 to BM
    - backward: STE

    ---
    - `width`: the number of bits (1 sign bit + exponent_bits + mantissa_bits)
    - `exponent_width`: the number of exponent_bits
    - `exponent_bias_width`: the number of bits of the shared exponent bias
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    return BlockMinifloatQuantize.apply(
        x,
        width,
        exponent_width,
        exponent_bias_width,
        block_size,
        skip_first_dim,
    )
