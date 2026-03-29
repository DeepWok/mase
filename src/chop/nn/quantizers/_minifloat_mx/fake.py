"""
Fake minifloat quantization operations.
"""

import torch
from torch import Tensor

from .meta import MinifloatMeta


def extract_minifloat_component(x: Tensor, minifloat_meta: MinifloatMeta) -> Tensor:
    """
    Extract minifloat representation from float tensor.

    Args:
        x: Input float tensor
        minifloat_meta: Minifloat format specification

    Returns:
        Tensor of uint16 containing minifloat representation
    """
    y_exp_bits = minifloat_meta.exp_bits
    y_frac_bits = minifloat_meta.frac_bits
    always_finite = minifloat_meta.is_finite
    round_mode = minifloat_meta.round_mode

    y_exp_bias = (1 << (y_exp_bits - 1)) - 1
    y_exp_max = (1 << y_exp_bits) - 1 if always_finite else (1 << y_exp_bits) - 2
    y_exp_max_biased = y_exp_max - y_exp_bias
    y_exp_min = 0
    y_exp_min_biased = y_exp_min - y_exp_bias
    y_frac_max = (1 << y_frac_bits) - 1

    x = x.to(torch.float32)
    y_sign = x < 0
    x_int32 = x.abs().view(torch.int32)
    flush_to_zero = (x_int32 & 0x7F800000) == 0
    x_normal = torch.where(flush_to_zero, 0.0, x)

    x_frac, x_exp = x_normal.abs().frexp()
    x_frac = x_frac * 2
    x_exp = x_exp - 1

    if not always_finite:
        x_is_inf = x.isinf()
        x_is_nan = x.isnan()

    y_exp = x_exp
    underflow = y_exp < y_exp_min_biased
    overflow = y_exp > y_exp_max_biased
    y_exp = y_exp + y_exp_bias

    y_frac = x_frac.view(torch.int32) & 0x7FFFFF

    if round_mode == "rz":
        y_frac = y_frac >> (23 - y_frac_bits)
    else:
        y_frac = (y_frac >> 8).float()
        div = 1 << (15 - y_frac_bits)
        y_frac = y_frac / div
        if round_mode == "ru":
            y_frac = y_frac.ceil()
        elif round_mode == "rd":
            y_frac = y_frac.floor()
        elif round_mode == "rn":
            y_frac = y_frac.round()
        else:
            raise ValueError(f"Unknown rounding mode: {round_mode}")
        y_frac = y_frac.to(torch.int32)

    y_is_subnormal = (y_exp == y_exp_min) & (y_frac != 0)
    y_frac = torch.where(y_is_subnormal, (y_frac | (1 << y_frac_bits)) >> 1, y_frac)

    # underflow -> 0
    y_frac = torch.where(underflow, 0, y_frac)
    y_exp = torch.where(underflow, 0, y_exp)
    # overflow -> max
    y_frac = torch.where(overflow, y_frac_max, y_frac)
    y_exp = torch.where(overflow, y_exp_max, y_exp)
    # flush to zero
    y_frac = torch.where(flush_to_zero, 0, y_frac)
    y_exp = torch.where(flush_to_zero, 0, y_exp)

    if not always_finite:
        y_frac = torch.where(x_is_inf, 0, y_frac)
        y_frac = torch.where(x_is_nan, (1 << y_frac_bits) - 1, y_frac)
        y_exp = torch.where(x_is_inf, y_exp_max, y_exp)
        y_exp = torch.where(x_is_nan, y_exp_max, y_exp)

    y = (y_exp << y_frac_bits) | y_frac
    y = torch.where(y_sign, y + (1 << (y_exp_bits + y_frac_bits)), y)
    y = y.to(torch.uint16)
    return y


def compose_minifloat_component(
    elements: Tensor,
    minifloat_meta: MinifloatMeta,
    output_dtype: torch.dtype,
) -> Tensor:
    """
    Compose float tensor from minifloat representation.

    Args:
        elements: Tensor of uint16 containing minifloat representation
        minifloat_meta: Minifloat format specification
        output_dtype: Desired output dtype

    Returns:
        Dequantized float tensor
    """
    exp_bits = minifloat_meta.exp_bits
    frac_bits = minifloat_meta.frac_bits
    always_finite = minifloat_meta.is_finite

    x_sign_mask = 1 << (exp_bits + frac_bits)
    x_frac_mask = (1 << frac_bits) - 1
    x_exp_bias = (1 << (exp_bits - 1)) - 1

    assert elements.dtype == torch.uint16
    elements = elements.to(torch.int32)
    y_sign = (elements & x_sign_mask) << (31 - (exp_bits + frac_bits))

    elements = elements & 0x7FFF
    x_exp = (elements >> frac_bits) & ((1 << exp_bits) - 1)
    x_frac = elements & x_frac_mask
    is_subnormal = (x_exp == 0) & (x_frac != 0)
    is_zero = (x_exp == 0) & (x_frac == 0)

    if not always_finite:
        y_is_not_finite = x_exp == ((1 << exp_bits) - 1)
        y_is_inf = y_is_not_finite & (x_frac == 0)
        y_is_nan = y_is_not_finite & (x_frac != 0)

    y_exp = x_exp - x_exp_bias
    y_exp = torch.where(is_subnormal, y_exp + 1, y_exp)
    y_exp = torch.exp2(y_exp)
    y_frac = x_frac.to(torch.float32)
    y_frac = y_frac / (1 << frac_bits)
    y_frac = torch.where(is_subnormal, y_frac, y_frac + 1.0)
    y = y_exp * y_frac

    if not always_finite:
        y = torch.where(y_is_inf, float("inf"), y)
        y = torch.where(y_is_nan, float("nan"), y)
    y = torch.where(is_zero, 0.0, y)
    y = torch.where(y_sign != 0, -y, y)
    y = y.to(output_dtype)
    return y
