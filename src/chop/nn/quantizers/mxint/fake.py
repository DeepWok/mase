"""
Fake MXINT quantization operations.
"""

import torch
from torch import Tensor

from .meta import MXIntMeta


def mxint_shared_exponent(maximum: Tensor, element_bits: int) -> Tensor:
    """Return the smallest exponent whose sign-magnitude range covers the block."""
    magnitude_levels = 2 ** (element_bits - 1)
    qmax = (magnitude_levels - 1) / magnitude_levels
    return (maximum.to(torch.float32) / qmax).log2().ceil()


def mxint_sign_magnitude_codes(elements: Tensor, element_bits: int) -> Tensor:
    """Encode signed integer elements as canonical MXINT sign-magnitude codes."""
    if element_bits not in (2, 4, 8):
        raise ValueError("MXINT element width must be 2, 4, or 8")
    integral = elements.round()
    if not torch.equal(elements, integral):
        raise ValueError("MXINT elements must be integral")
    magnitude = integral.abs().to(torch.int64)
    magnitude_max = 2 ** (element_bits - 1) - 1
    if torch.any(magnitude > magnitude_max):
        raise ValueError("MXINT magnitude is outside its declared width")
    sign = ((integral < 0) & (magnitude != 0)).to(torch.int64)
    return (sign << (element_bits - 1)) | magnitude


def extract_mxint_components(
    x: Tensor, mxint_meta: MXIntMeta, percentile: float = 1.0
) -> tuple[Tensor, Tensor]:
    """
    Extract MXINT components (scale and elements) from a tensor.

    Args:
        x: Input tensor (already flattened)
        mxint_meta: MXINT format specification
        percentile: Percentile for scale calculation (1.0 = max)

    Returns:
        Tuple of (scale, quantized_mantissa)
    """
    B = mxint_meta.block_size
    assert (
        x.numel() % B == 0
    ), f"Input tensor size {x.numel()} is not divisible by block size {B}."
    n_blocks = x.numel() // B

    x = x.flatten()
    x = x.reshape(n_blocks, B)

    ori_dtype = x.dtype
    # quantile needs fp32; at percentile 1.0 it reduces to the block maximum,
    # which avoids sorting every block and quantile's input-size limit.
    magnitude = x.abs().to(torch.float32)
    x_max = (
        magnitude.amax(dim=1, keepdim=True)
        if percentile == 1.0
        else magnitude.quantile(percentile, dim=1, keepdim=True)
    ).to(ori_dtype)

    zero_blocks = x_max == 0
    scale_bias = 2 ** (mxint_meta.scale_bits - 1) - 1
    scale_min = -scale_bias
    scale_max = 2**mxint_meta.scale_bits - 1 - scale_bias
    unit_max = torch.where(zero_blocks, torch.ones_like(x_max), x_max)
    scale = mxint_shared_exponent(
        unit_max,
        mxint_meta.element_bits,
    ).clamp(min=scale_min, max=scale_max)
    scale = torch.where(zero_blocks, torch.zeros_like(scale), scale)
    x = x / 2**scale
    x_mant = x * 2 ** (mxint_meta.element_bits - 1)
    magnitude_max = 2 ** (mxint_meta.element_bits - 1) - 1
    x_mant = x_mant.round().clamp(
        min=-magnitude_max,
        max=magnitude_max,
    )
    scale = scale + scale_bias

    return scale, x_mant


def compose_mxint_tensor(
    shared_scales: Tensor,
    elements: Tensor,
    mxint_meta: MXIntMeta,
) -> Tensor:
    """
    Compose tensor from MXINT components.

    Args:
        shared_scales: Shared scales tensor
        elements: Quantized elements tensor
        mxint_meta: MXINT format specification

    Returns:
        Dequantized tensor
    """
    scale_bias = 2 ** (mxint_meta.scale_bits - 1) - 1
    return (
        elements
        / 2 ** (mxint_meta.element_bits - 1)
        * 2 ** (shared_scales - scale_bias)
    )
