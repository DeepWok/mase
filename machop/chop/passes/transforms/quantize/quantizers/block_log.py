from typing import Any, Union

import torch
from numpy import ndarray
from torch import Tensor
from .log import log_quantizer

from .utils import block, my_clamp, unblock


def _block_log_quantize(
    x: Union[Tensor, ndarray],
    width: int,
    exponent_bias_width: int = None,
    block_size: int = 16,
    skip_first_dim: bool = False,
):
    """
    Convert IEEE FP32/64 to block base-2 log quantized values. A bias is shared over each block

    ---
    - forward: convert IEEE FP32/64 to base-2 log quantized values
    - backward: This is not STE but close to STE because the derivate of (2**exponent) depends on the rounded exponent

    ---
    - `width`: the number of bits, including 1 sign bit and (bits-1) exponent bits
    - `exponent_bias_width`: the number of bits of shared exponent bias
    - `block_size`: a list of integers where each integer is the block size along the corresponding dim

    """
    exponent_bits = width - 1
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )

    # fill zeros to avoid log2(0) = -inf
    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    per_block_max_exponent = torch.ceil(torch.log2(per_block_max))
    per_block_bias = my_clamp(
        2**exponent_bits - 1 - per_block_max_exponent, 0, 2**exponent_bias_width - 1
    )

    per_block_lq_x = log_quantizer(blocked_x, width=width, exponent_bias=per_block_bias)
    lq_x = unblock(
        per_block_lq_x,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )

    return lq_x


class BlockLogQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        width: int,
        exponent_bias_width: int = None,
        block_size: int = 16,
        skip_first_dim: bool = False,
    ):
        return _block_log_quantize(
            x,
            width=width,
            exponent_bias_width=exponent_bias_width,
            block_size=block_size,
            skip_first_dim=skip_first_dim,
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def block_log_quantizer(
    x: Tensor,
    width: int,
    exponent_bias_width: int = None,
    block_size: int = 16,
    skip_first_dim: bool = False,
):
    """
    Convert IEEE FP32/64 to block base-2 log quantized values. A bias is shared over each block

    ---
    - forward: convert IEEE FP32/64 to base-2 log quantized values
    - backward: This is not STE but close to STE because the derivate of (2**exponent) depends on the rounded exponent

    ---
    - `width`: the number of bits, including 1 sign bit and (bits-1) exponent bits
    - `exponent_bias_width`: the number of bits of shared exponent bias
    - `block_size`: a list of integers where each integer is the block size along the corresponding dim
    """
    return BlockLogQuantize.apply(
        x,
        width,
        exponent_bias_width,
        block_size,
        skip_first_dim,
    )
