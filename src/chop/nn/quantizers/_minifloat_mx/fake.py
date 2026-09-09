"""
Fake minifloat quantization operations.
"""

import torch
from torch import Tensor

from .meta import MinifloatMeta


def _minifloat_fields(
    x: Tensor, minifloat_meta: MinifloatMeta
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (sign, biased exponent, fraction) for one minifloat rounding."""
    y_exp_bits = minifloat_meta.exp_bits
    y_frac_bits = minifloat_meta.frac_bits
    always_finite = minifloat_meta.is_finite
    round_mode = minifloat_meta.round_mode

    y_exp_bias = 1 if y_exp_bits == 1 else (1 << (y_exp_bits - 1)) - 1
    y_exp_max = (1 << y_exp_bits) - 1 if always_finite else (1 << y_exp_bits) - 2
    y_exp_max_biased = y_exp_max - y_exp_bias
    y_exp_min_biased = 1 - y_exp_bias
    y_frac_max = (1 << y_frac_bits) - 1
    y_frac_levels = 1 << y_frac_bits

    x = x.to(torch.float32)
    y_sign = torch.signbit(x)
    magnitude = x.abs()
    nonzero_finite = torch.isfinite(magnitude) & (magnitude != 0)

    def round_magnitude(value: Tensor) -> Tensor:
        if round_mode == "rn":
            return value.round()
        if round_mode == "rz":
            return value.floor()
        if round_mode == "ru":
            return torch.where(y_sign, value.floor(), value.ceil())
        if round_mode == "rd":
            return torch.where(y_sign, value.ceil(), value.floor())
        raise ValueError(f"Unknown rounding mode: {round_mode}")

    min_normal = float(2**y_exp_min_biased)
    subnormal_step = min_normal / y_frac_levels
    is_subnormal = nonzero_finite & (magnitude < min_normal)

    safe_magnitude = torch.where(
        nonzero_finite,
        magnitude,
        torch.ones_like(magnitude),
    )
    unbiased_exp = torch.floor(torch.log2(safe_magnitude))
    overflow = (~torch.isfinite(magnitude)) | (
        nonzero_finite & (unbiased_exp > y_exp_max_biased)
    )
    clamped_exp = unbiased_exp.clamp(
        min=y_exp_min_biased,
        max=y_exp_max_biased,
    )
    normal_fraction = (
        safe_magnitude / torch.exp2(clamped_exp) - 1.0
    ) * y_frac_levels
    y_frac = round_magnitude(normal_fraction).to(torch.int32)
    y_exp = (clamped_exp + y_exp_bias).to(torch.int32)

    carry = y_frac >= y_frac_levels
    y_frac = torch.where(carry, 0, y_frac)
    y_exp = torch.where(carry, y_exp + 1, y_exp)

    subnormal_fraction = round_magnitude(
        magnitude / subnormal_step
    ).to(torch.int32)
    subnormal_carry = subnormal_fraction >= y_frac_levels
    y_frac = torch.where(
        is_subnormal,
        torch.where(subnormal_carry, 0, subnormal_fraction),
        y_frac,
    )
    y_exp = torch.where(
        is_subnormal,
        torch.where(subnormal_carry, 1, 0),
        y_exp,
    )

    overflow = overflow | (y_exp > y_exp_max)
    y_frac = torch.where(overflow, y_frac_max, y_frac)
    y_exp = torch.where(overflow, y_exp_max, y_exp)
    y_frac = torch.where(nonzero_finite | overflow, y_frac, 0)
    y_exp = torch.where(nonzero_finite | overflow, y_exp, 0)
    return y_sign, y_exp, y_frac


def extract_minifloat_component(x: Tensor, minifloat_meta: MinifloatMeta) -> Tensor:
    """
    Extract minifloat representation from float tensor.

    Args:
        x: Input float tensor
        minifloat_meta: Minifloat format specification

    Returns:
        Tensor of uint16 containing minifloat representation
    """
    y_sign, y_exp, y_frac = _minifloat_fields(x, minifloat_meta)
    y_frac_bits = minifloat_meta.frac_bits
    y = (y_exp << y_frac_bits) | y_frac
    y = torch.where(y_sign, y + (1 << (minifloat_meta.exp_bits + y_frac_bits)), y)
    return y.to(torch.uint16)


def _quantize_minifloat_value(x: Tensor, minifloat_meta: MinifloatMeta) -> Tensor:
    """Round to the minifloat grid and return the value, skipping the encoding.

    Equivalent to ``compose_minifloat_component(extract_minifloat_component(x))``
    but without the uint16 round trip. The encoder never emits an inf/nan
    exponent code, so the decoder's special-value branches cannot apply here.
    """
    y_sign, y_exp, y_frac = _minifloat_fields(x, minifloat_meta)
    y_exp_bias = 1 if minifloat_meta.exp_bits == 1 else (
        (1 << (minifloat_meta.exp_bits - 1)) - 1
    )
    frac_levels = 1 << minifloat_meta.frac_bits
    subnormal_step = float(2 ** (1 - y_exp_bias)) / frac_levels
    fraction = y_frac.to(torch.float32)
    magnitude = torch.where(
        y_exp == 0,
        fraction * subnormal_step,
        (1.0 + fraction / frac_levels)
        * torch.exp2((y_exp - y_exp_bias).to(torch.float32)),
    )
    return torch.where(y_sign, -magnitude, magnitude)


from chop.nn.quantizers._compile import ENABLED as _COMPILE_ENABLED  # noqa: E402
from chop.nn.quantizers._compile import compiled_variant  # noqa: E402


def quantize_minifloat_value(x, minifloat_meta):
    """Compiled per-format variant; constants live in the closure."""

    if not _COMPILE_ENABLED:
        return _quantize_minifloat_value(x, minifloat_meta)
    meta = minifloat_meta
    key = ("minifloat", meta.exp_bits, meta.frac_bits, meta.is_finite, meta.round_mode)
    return compiled_variant(key, lambda: (lambda t: _quantize_minifloat_value(t, meta)))(x)


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
    x_exp_bias = 1 if exp_bits == 1 else (1 << (exp_bits - 1)) - 1

    assert elements.dtype == torch.uint16
    elements = elements.to(torch.int32)
    y_sign = (elements & x_sign_mask) << (31 - (exp_bits + frac_bits))

    elements = elements & (x_sign_mask - 1)
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
