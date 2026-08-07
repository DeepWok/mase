"""PLENA MXFP quantize-dequantize helpers."""

import torch
from torch import Tensor

from .._minifloat_mx import minifloat_quantizer_sim
from .meta import MXFPMeta


def extract_mxfp_components(
    x: Tensor, mxfp_meta: MXFPMeta, percentile: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Split a blocked tensor into shared scales and minifloat elements.

    Returns:
        scales:   per-block shared exponent in log2 domain (float), shape
                  ``[n_blocks, 1]``. Recompose with ``2 ** scales``.
        elements: per-element minifloat values in block-relative space,
                  shape ``[n_blocks, B]``.
    """
    B = mxfp_meta.block_size
    assert (
        x.numel() % B == 0
    ), f"Input tensor size {x.numel()} is not divisible by block size {B}."
    n_blocks = x.numel() // B

    x = x.flatten().reshape(n_blocks, B)
    magnitude = x.abs().to(torch.float32)
    # quantile() sorts each block; at percentile 1.0 the result is the maximum,
    # which is also the only path that avoids quantile's input-size limit.
    per_block_max = (
        magnitude.amax(dim=1, keepdim=True)
        if percentile == 1.0
        else magnitude.quantile(percentile, dim=1, keepdim=True)
    )
    nonzero_block = per_block_max > 0
    safe_block_max = torch.where(
        nonzero_block,
        per_block_max,
        torch.ones_like(per_block_max),
    )
    # OCP MX shared exponent: place the block maximum at the top of the
    # element format's exponent range, so the elements use their full range.
    scales = safe_block_max.log2().floor() - mxfp_meta.element_max_exponent
    scale_bias = 2 ** (mxfp_meta.scale_exp_bits - 1) - 1
    scales = scales.clamp(
        min=-scale_bias,
        max=2**mxfp_meta.scale_exp_bits - 1 - scale_bias,
    )
    scales = torch.where(nonzero_block, scales, torch.zeros_like(scales))

    q_tensor = x / 2**scales
    elements = minifloat_quantizer_sim(
        q_tensor,
        minifloat_meta=mxfp_meta.element_meta,
        output_dtype=x.dtype,
    )

    return scales, elements


def compose_mxfp_tensor(
    shared_scales: Tensor,
    elements: Tensor,
    mxfp_meta: MXFPMeta,
    output_dtype: torch.dtype,
) -> Tensor:
    """Reconstruct the dequantized tensor from MXFP components.

    ``elements`` are block-relative minifloat values; the shared log2 scale
    shifts them back into the original block magnitude.
    """
    del mxfp_meta  # shape and format already fixed by extraction
    dequantized = (elements * 2**shared_scales).flatten().to(output_dtype)
    return dequantized
