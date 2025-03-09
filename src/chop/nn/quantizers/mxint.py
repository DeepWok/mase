import torch
from torch import Tensor

from .utils import block, my_clamp, my_round, unblock


def _mxint_quantize(
    x: Tensor,
    width: int = 8,
    exponent_width: int = 4,
    block_size: list[int] = [16],
    skip_first_dim: bool = True,
):
    """
    - Convert IEEE FP32/64 to Microscaling Interger (MXINT), where an exponent is shared over all elements in a block.
    - https://arxiv.org/pdf/2310.10537.pdf

    ---
    - forward: convert IEEE FP32/64 to MXINT
    - backward: STE

    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.
    """

    if isinstance(block_size, int):
        block_size = [block_size]

    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )

    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    exponent_bias = 2 ** (exponent_width - 1) - 1

    per_block_scale = torch.floor(torch.log2(per_block_max))
    per_block_scale = my_clamp(per_block_scale, -exponent_bias, exponent_bias)

    # OCP MX INT Quantization
    sign = torch.sign(blocked_x + 1e-9)
    value = torch.abs(blocked_x) + 1e-9

    scaled_value = value / (2**per_block_scale)

    element_max = 2 ** (width - 1) - 1
    shift = 2 ** (width - 2)
    # To advoid introducing a negative bias

    quantized_value = my_clamp(my_round(scaled_value * shift), 0, element_max)

    element_value = quantized_value / shift

    mxint_value = sign * element_value * (2**per_block_scale)

    mxint_x = unblock(
        mxint_value,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )

    # fmt: off
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(x, torch.tensor([0.0], dtype=x.dtype, device=x.device))
    mxint_x = (~is_close_to_0) * mxint_x + (is_close_to_0) * x
    # fmt: on

    return mxint_x


class MXINTQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        width: int = 8,
        exponent_width: int = 4,
        block_size: list[int] = [16],
        skip_first_dim: bool = True,
    ):
        return _mxint_quantize(
            x,
            width=width,
            exponent_width=exponent_width,
            block_size=block_size,
            skip_first_dim=skip_first_dim,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


def mxint_quantizer(
    x: Tensor,
    width: int = 8,
    exponent_width: int = 4,
    block_size: list[int] = [16],
    skip_first_dim: bool = True,
):
    return MXINTQuantize.apply(
        x,
        width,
        exponent_width,
        block_size,
        skip_first_dim,
    )
