import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# from .quantizers import integer_quantizer
from .utils import block, my_clamp, my_round, unblock, my_floor


def integer_quantizer_for_hw(x: Tensor, width: int, frac_width: int):
    thresh = 2 ** (width - 1)
    scale = 2**frac_width

    fixed_point_value = my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1)
    fixed_point_value = fixed_point_value.to(torch.int)
    fixed_point_value = fixed_point_value % (2**width)
    return fixed_point_value


def unsigned_integer_quantizer_for_hw(x: Tensor, width: int, frac_width: int):
    thresh = 2**width - 1
    scale = 2**frac_width

    fixed_point_value = my_clamp(my_floor(x.mul(scale)), 0, thresh)
    fixed_point_value = fixed_point_value.to(torch.int)
    fixed_point_value = fixed_point_value % (2**width)
    return fixed_point_value


def integer_floor_quantizer_for_hw(x: Tensor, width: int, frac_width: int):
    thresh = 2 ** (width - 1)
    scale = 2**frac_width

    fixed_point_value = my_clamp(my_floor(x.mul(scale)), -thresh, thresh - 1)
    fixed_point_value = fixed_point_value.to(torch.int)
    fixed_point_value = fixed_point_value % (2**width)
    return fixed_point_value


def mxint_quantizer_for_hw(
    x: Tensor,
    width: int,
    exponent_width: int,
    block_size: list[int],
    floor: bool = False,
):
    """
    - Convert IEEE FP32/64 to Microscaling Interger (MXINT), where an exponent is shared over all elements in a block.
    - https://arxiv.org/pdf/2310.10537.pdf
    - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf


    ---
    - forward: convert IEEE FP32/64 to MXINT
    - backward: STE

    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.
    """

    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x,
        block_shape=block_size,
    )

    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    exponent_bias = 2 ** (exponent_width - 1) - 1

    per_block_expontent = my_floor(torch.log2(per_block_max)) + exponent_bias
    per_block_expontent = my_clamp(per_block_expontent, 0, 2**exponent_width - 1)

    element_max = 2 ** (width - 1) - 1
    shift = 2 ** (width - 2)

    scaled_value = shift * blocked_x / 2 ** (per_block_expontent - exponent_bias)

    if floor:
        quantized_value = my_floor(scaled_value)
    else:
        quantized_value = my_round(scaled_value)

    quantized_value = my_clamp(quantized_value, -element_max, element_max)

    return quantized_value, per_block_expontent


# sw_quantizer_to_hw_quantizer = {integer_quantizer: integer_quantizer_for_hw}
