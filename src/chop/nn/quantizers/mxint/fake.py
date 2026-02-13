"""
Fake MXINT quantization operations.
"""

import torch
from torch import Tensor

from .meta import MXIntMeta


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
    assert x.numel() % B == 0, (
        f"Input tensor size {x.numel()} is not divisible by block size {B}."
    )
    n_blocks = x.numel() // B

    x = x.flatten()
    x = x.reshape(n_blocks, B)

    ori_dtype = x.dtype
    # quantile needs fp32
    x_max = x.abs().to(torch.float32).quantile(percentile, dim=1, keepdim=True).to(ori_dtype)

    scale = x_max.log2().ceil()
    scale_bias = 2 ** (mxint_meta.scale_bits - 1) - 1
    x = x / 2 ** scale
    x_mant = x * 2 ** (mxint_meta.element_bits - 1)
    scale = scale + scale_bias
    scale = scale.clamp(min=0, max=2 ** mxint_meta.scale_bits - 1)
    x_mant = x_mant.round().clamp(
        min=-2 ** (mxint_meta.element_bits - 1),
        max=2 ** (mxint_meta.element_bits - 1) - 1
    )

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
    return elements / 2 ** (mxint_meta.element_bits - 1) * 2 ** (shared_scales - scale_bias)
