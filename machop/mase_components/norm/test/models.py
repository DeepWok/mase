import logging
from functools import partial
from math import ceil, log2

import torch
from torch import Tensor

from chop.passes.graph.transforms.quantize.quantizers.integer import (
    integer_floor_quantizer,
)
from chop.passes.graph.transforms.quantize.quantizers.quantizers_for_hw import (
    integer_floor_quantizer_for_hw,
)

from mase_components.cast.test.fixed_signed_cast_tb import _fixed_signed_cast_model
from mase_components.fixed_math.test.isqrt_sw import isqrt_sw2


logger = logging.getLogger("norm.models")
logger.setLevel(logging.INFO)


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

    squares = diff**2
    logger.debug("Squares:")
    logger.debug(squares[0])
    squares_int = (squares * (2**square_frac_width)).int()
    logger.debug(squares * (2**square_frac_width))

    sum_squares = torch.sum(squares, dim=(1, 2, 3), keepdim=True)
    sum_squares = integer_floor_quantizer(
        sum_squares, variance_width, variance_frac_width
    )
    sum_squares_int = integer_floor_quantizer_for_hw(
        sum_squares.clone(), variance_width, variance_frac_width
    )

    num_vals = x.shape[1] * x.shape[2] * x.shape[3]
    logger.debug("Num Values: %d" % (num_vals))
    var = sum_squares / num_vals
    var = integer_floor_quantizer(var, variance_width, variance_frac_width)
    var_i = integer_floor_quantizer_for_hw(
        var.clone(), variance_width, variance_frac_width
    )
    logger.debug("Variance:")
    logger.debug(f"{var[0]}")

    # Clamp down variance to isqrt width
    var_clamp = torch.clamp(var, 0.0, ((2**isqrt_width) - 1) / (2**isqrt_frac_width))
    logger.debug("Variance Clamped:")
    logger.debug(f"{var_clamp[0]}")
    var_clamp_int = (var_clamp * (2**isqrt_frac_width)).int()

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

    inv_sqrt = inv_sqrt_int / (2**isqrt_frac_width)
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
        norm_out, out_width, out_frac_width, symmetric=False, rounding_mode="floor"
    )
    logger.debug("Norm (Casted):")
    logger.debug(norm_out_float[0])

    return (
        norm_out_float,
        norm_int_out,
        {
            "mu": mu_int.squeeze(dim=(1, 2, 3)).tolist(),
            "squares": squares_int,
            "sum_squares": sum_squares_int.squeeze(dim=(1, 2, 3)).tolist(),
            "var": var_i.squeeze(dim=(1, 2, 3)).tolist(),
            "var_clamp": var_clamp_int.squeeze(dim=(1, 2, 3)).tolist(),
            "isqrt": inv_sqrt_int.squeeze(dim=(1, 2, 3)).tolist(),
            "diff": diff_int,
            "norm": norm_int,
        },
    )


def _fixed_rms_norm_2d_model(
    x: Tensor,
    acc_width: int,
    acc_frac_width: int,
    inv_sqrt_width: int,
    inv_sqrt_frac_width: int,
    out_width: int,
    out_frac_width: int,
):
    logger.debug("Input:")
    logger.debug(x[0])

    # Sum of Squares
    sum_sq = torch.square(x).sum(dim=(1, 2, 3), keepdim=True)
    sum_sq = integer_floor_quantizer(sum_sq, acc_width, acc_frac_width)
    logger.debug("Sum of Squares:")
    logger.debug(sum_sq[0])

    # Divide to get mean square
    mean_sq = sum_sq / (x.shape[1] * x.shape[2] * x.shape[3])
    mean_sq = integer_floor_quantizer(mean_sq, acc_width, acc_frac_width)
    logger.debug("Mean Square:")
    logger.debug(mean_sq[0])

    # Get inverse sqrt of mean square
    # inv_sqrt = inv_sqrt_model(mean_sq)  # TODO: Add inv sqrt model
    inv_sqrt = torch.full_like(mean_sq, 0.25)  # TODO: remove this later
    inv_sqrt = integer_floor_quantizer(inv_sqrt, inv_sqrt_width, inv_sqrt_frac_width)
    logger.debug("Inverse SQRT:")
    logger.debug(inv_sqrt[0])

    # Norm calculation
    norm_out = x * inv_sqrt
    logger.debug("Norm:")
    logger.debug(norm_out[0])
    norm_out_float, norm_int_out = _fixed_signed_cast_model(
        norm_out, out_width, out_frac_width, symmetric=False, rounding_mode="floor"
    )

    return norm_out_float, norm_int_out
