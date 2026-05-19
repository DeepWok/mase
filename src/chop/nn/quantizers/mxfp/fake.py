"""
Fake MXFP quantization — aligned with Plena-Acc-Sim.

"""

import torch
from torch import Tensor

from ..minifloat import minifloat_denorm_quantizer
from .meta import MXFPMeta


def extract_mxfp_components(
    x: Tensor, mxfp_meta: MXFPMeta, percentile: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Extract MXFP components (Plena-aligned).

    Returns:
        scales:   per-block shared exponent in **log2 domain** (float),
                  shape ``[n_blocks, 1]``. Recompose with ``2 ** scales``.
        elements: per-element minifloat-quantized values (float, same dtype
                  as ``x``), shape ``[n_blocks, B]``. Already in scaled
                  (block-relative) space — multiply by ``2 ** scales`` to
                  get the dequantized tensor.
    """
    B = mxfp_meta.block_size
    assert x.numel() % B == 0, (
        f"Input tensor size {x.numel()} is not divisible by block size {B}."
    )
    n_blocks = x.numel() // B

    x = x.flatten().reshape(n_blocks, B)
    per_block_max = (
        x.abs().to(torch.float32).quantile(percentile, dim=1, keepdim=True) + 1e-9
    )
    scales = per_block_max.log2().ceil()
    scales = scales.clamp(
        min=-(2 ** (mxfp_meta.scale_exp_bits - 1)),
        max=2 ** (mxfp_meta.scale_exp_bits - 1) - 1,
    )

    q_tensor = x / 2**scales
    elements = minifloat_denorm_quantizer(
        q_tensor,
        width=mxfp_meta.element_frac_bits + mxfp_meta.element_exp_bits + 1,
        exponent_width=mxfp_meta.element_exp_bits,
    )

    return scales, elements


def compose_mxfp_tensor(
    shared_scales: Tensor,
    elements: Tensor,
    mxfp_meta: MXFPMeta,
    output_dtype: torch.dtype,
) -> Tensor:
    """Reconstruct the dequantized tensor from Plena-style MXFP components.

    ``elements`` are the already-block-relative minifloat values; the shared
    log2 scale just shifts them back into the original block magnitude.
    ``mxfp_meta`` is unused for the recompose itself (kept in the signature
    so callers don't need to know whether the impl is Plena- or OCP-style).
    """
    del mxfp_meta  # only needed by the OCP-style impl
    dequantized = (elements * 2**shared_scales).flatten().to(output_dtype)
    return dequantized
