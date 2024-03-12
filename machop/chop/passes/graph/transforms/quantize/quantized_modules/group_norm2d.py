import logging
from math import ceil, log2
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from ..quantizers.integer import integer_floor_quantizer
from ..quantizers.quantizers_for_hw import integer_floor_quantizer_for_hw
from .fixed_signed_cast import _fixed_signed_cast_model


from mase_components.fixed_arithmetic.test.isqrt_sw import (
    isqrt_sw2
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def _fixed_group_norm_2d_model(
    x: Tensor,
    in_width: int,
    in_frac_width: int,
    diff_width: int,
    diff_frac_width: int,
    square_width: int,
    square_frac_width: int,
    variance_width: int,
    variance_frac_width: int,
    isqrt_width: int,
    isqrt_frac_width: int,
    isqrt_lut: list,
    norm_width: int,
    norm_frac_width: int,
    out_width: int,
    out_frac_width: int,
):
    logger.debug("Input:")
    logger.debug(x[0])

    # Mean calculation
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    logger.debug("Mu:")
    logger.debug(mu[0])
    mu = integer_floor_quantizer(mu, in_width, in_frac_width)
    mu_int = integer_floor_quantizer_for_hw(mu.clone(), in_width, in_frac_width)
    logger.debug("Mu Quantized:")
    logger.debug(mu[0])

    # Variance calculation
    diff = x - mu
    diff_int = integer_floor_quantizer_for_hw(diff.clone(), diff_width, diff_frac_width)
    logger.debug("Diff:")
    logger.debug(diff[0])

    squares = diff ** 2
    logger.debug("Squares:")
    logger.debug(squares[0])
    squares_int = (squares * (2**square_frac_width)).int()
    logger.debug(squares * (2**square_frac_width))

    sum_squares = torch.sum(squares, dim=(1, 2, 3), keepdim=True)
    sum_squares = integer_floor_quantizer(sum_squares, variance_width, variance_frac_width)
    sum_squares_int = integer_floor_quantizer_for_hw(sum_squares.clone(), variance_width, variance_frac_width)

    num_vals = x.shape[1] * x.shape[2] * x.shape[3]
    logger.debug("Num Values: %d" % (num_vals))
    var = sum_squares / num_vals
    var = integer_floor_quantizer(var, variance_width, variance_frac_width)
    var_i = integer_floor_quantizer_for_hw(var.clone(), variance_width, variance_frac_width)
    logger.debug("Variance:")
    logger.debug(f"{var[0]}")

    # Clamp down variance to isqrt width
    var_clamp = torch.clamp(var, 0.0, ((2**isqrt_width)-1)/(2**isqrt_frac_width))
    logger.debug("Variance Clamped:")
    logger.debug(f"{var_clamp[0]}")
    var_clamp_int = (var_clamp * (2 ** isqrt_frac_width)).int()

    # Inverse Square Root calculation
    lut_pow = ceil(log2(len(isqrt_lut)))
    logger.debug("Variance INT:")
    logger.debug(f"{var_clamp_int[0]}")

    f = partial(
        isqrt_sw2,
        in_width=isqrt_width,
        frac_width=isqrt_frac_width,
        lut_pow=lut_pow,
        lut=isqrt_lut,
        debug=False,
    )
    inv_sqrt_int = var_clamp_int.clone().apply_(f)

    logger.debug("INV SQRT INT:")
    logger.debug(f"{inv_sqrt_int[0]}")

    inv_sqrt = inv_sqrt_int / (2 ** isqrt_frac_width)
    logger.debug("Inverse SQRT:")
    logger.debug(f"{inv_sqrt[0]}")

    # Norm calculation
    norm_out = diff * inv_sqrt
    norm_int = integer_floor_quantizer_for_hw(
        norm_out.clone(), norm_width, norm_frac_width
    )
    logger.debug("Norm:")
    logger.debug(norm_out[0])

    norm_out_float, norm_int_out = _fixed_signed_cast_model(
        norm_out, out_width, out_frac_width,
        symmetric=False, rounding_mode="floor"
    )
    logger.debug("Norm (Casted):")
    logger.debug(norm_out_float[0])

    return norm_out_float, norm_int_out, {
        "mu": mu_int.squeeze(dim=(1, 2, 3)).tolist(),
        "squares": squares_int,
        "sum_squares": sum_squares_int.squeeze(dim=(1, 2, 3)).tolist(),
        "var": var_i.squeeze(dim=(1, 2, 3)).tolist(),
        "var_clamp": var_clamp_int.squeeze(dim=(1, 2, 3)).tolist(),
        "isqrt": inv_sqrt_int.squeeze(dim=(1, 2, 3)).tolist(),
        "diff": diff_int,
        "norm": norm_int,
    }


class GroupNorm2dInteger(nn.Module):
    def __init__(
        self,
        in_width: int,
        in_frac_width: int,
        variance_width: int,
        variance_frac_width: int,
        inv_sqrt_width: int,
        inv_sqrt_frac_width: int,
        out_width: int,
        out_frac_width: int,
    ) -> None:
        super().__init__()

        self.in_width = in_width
        self.in_frac_width = in_frac_width
        self.variance_width = variance_width
        self.variance_frac_width = variance_frac_width
        self.inv_sqrt_width = inv_sqrt_width
        self.inv_sqrt_frac_width = inv_sqrt_frac_width
        self.out_width = out_width
        self.out_frac_width = out_frac_width

    def forward(self, x: Tensor):
        return _fixed_group_norm_2d_model(
            x=x,
            in_width=self.in_width,
            in_frac_width=self.in_frac_width,
            variance_width=self.variance_width,
            variance_frac_width=self.variance_frac_width,
            inv_sqrt_width=self.inv_sqrt_width,
            inv_sqrt_frac_width=self.inv_sqrt_frac_width,
            out_width=self.out_width,
            out_frac_width=self.out_frac_width,
        )
