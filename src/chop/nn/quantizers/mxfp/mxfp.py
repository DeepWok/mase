"""
MXFP quantizer.
"""

import torch
from torch import Tensor
from tqdm import tqdm

from .meta import MXFPMeta, MXFPTensorMeta
from .helpers import flatten_for_quantize, permute_for_dequantize
from .fake import extract_mxfp_components, compose_mxfp_tensor


def mxfp_quantizer_sim(
    tensor: Tensor,
    block_dim: int,
    mxfp_meta: MXFPMeta,
    act_tensor: Tensor | None = None,
    dtype: torch.dtype | None = None,
    quantile_search: bool = False,
    cali_batch_size: int = 32,
) -> Tensor:
    """
    Quantize and dequantize a tensor using MXFP format.

    Args:
        tensor: Input tensor to quantize
        block_dim: Dimension to apply block quantization
        mxfp_meta: MXFP format specification
        act_tensor: Optional activation tensor for GPTQ-style calibration
        dtype: Output dtype (default: same as input)
        quantile_search: Enable quantile-based clipping search
        cali_batch_size: Batch size for calibration

    Returns:
        Dequantized tensor
    """
    out_dq = torch.zeros_like(tensor)

    if quantile_search:
        qtensor = tensor.flatten()
        B = mxfp_meta.block_size

        qtensor = qtensor.reshape(-1, B)
        best = torch.full([qtensor.shape[0]], float('inf'), device=tensor.device, dtype=tensor.dtype)
        best_scales, best_elements, tensor_meta = _extract_with_meta(
            tensor, block_dim, mxfp_meta, percentile=1.0
        )

        percentiles = [1.0, 0.995, 0.99, 0.97, 0.95, 0.93, 0.90, 0.80, 0.70, 0.60, 0.50]
        for percentile in percentiles:
            scales, elements, tensor_meta = _extract_with_meta(
                tensor, block_dim, mxfp_meta, percentile=percentile
            )
            scale_bias = 2 ** (mxfp_meta.scale_exp_bits - 1) - 1
            q = elements / 2 ** (mxfp_meta.element_frac_bits - 1) * 2 ** (scales - scale_bias)
            q = q.to(dtype=qtensor.dtype)

            if act_tensor is not None:
                BATCH_SIZE = cali_batch_size
                last_dim = act_tensor.shape[-1]
                if last_dim != B:
                    assert last_dim % B == 0
                    act_tensor = act_tensor.view(*act_tensor.shape[:-1], last_dim // B, B)

                total_batches = act_tensor.shape[0]
                err = torch.zeros(qtensor.shape[0], device=tensor.device, dtype=tensor.dtype)

                with torch.no_grad():
                    for b in tqdm(range(0, total_batches, BATCH_SIZE), desc="Batching quant output", disable=True):
                        act_b = act_tensor[b:b + BATCH_SIZE]
                        out_q = torch.matmul(act_b, q.T)
                        out_orig = torch.matmul(act_b, qtensor.T)
                        err += torch.norm(out_q - out_orig, p=2, dim=(0, 1))

                        del act_b, out_q, out_orig
                        torch.cuda.empty_cache()

                torch.cuda.empty_cache()
            else:
                q -= qtensor
                q.abs_()
                q.pow_(2)
                err = torch.sum(q, 1)

            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                best_scales[tmp] = scales[tmp]
                best_elements[tmp] = elements[tmp]

    else:
        best_scales, best_elements, tensor_meta = _extract_with_meta(
            tensor, block_dim, mxfp_meta, percentile=1.0
        )

    out_dq = compose_mxfp_tensor(best_scales, best_elements, tensor_meta.meta, output_dtype=dtype or tensor.dtype)
    out_dq = permute_for_dequantize(out_dq, tensor_meta.shape, tensor_meta.block_dim)
    return out_dq


def _extract_with_meta(
    tensor: Tensor,
    block_dim: int,
    mxfp_meta: MXFPMeta,
    percentile: float = 1.0,
) -> tuple[Tensor, Tensor, MXFPTensorMeta]:
    """Extract MXFP components with tensor metadata."""
    device = str(tensor.device)
    ori_shape = tuple(tensor.shape)
    ori_dtype = str(tensor.dtype).removeprefix("torch.")
    ndim = len(ori_shape)
    assert block_dim < ndim and block_dim >= -ndim

    tensor_flat = flatten_for_quantize(tensor, block_dim)
    scales, elements = extract_mxfp_components(tensor_flat, mxfp_meta, percentile=percentile)

    tensor_meta = MXFPTensorMeta(
        device=device,
        dtype=ori_dtype,
        shape=ori_shape,
        block_dim=block_dim,
        meta=mxfp_meta,
    )
    return scales, elements, tensor_meta


# =============================================================================
# Mase-style quantizer interface with STE
# =============================================================================


class MXFPQuantize(torch.autograd.Function):
    """Autograd function for MXFP quantization with STE gradient."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        block_size: int,
        element_exp_bits: int,
        element_frac_bits: int,
        block_dim: int,
        scale_exp_bits: int,
        quantile_search: bool,
    ) -> Tensor:
        meta = MXFPMeta(
            block_size=block_size,
            scale_exp_bits=scale_exp_bits,
            element_exp_bits=element_exp_bits,
            element_frac_bits=element_frac_bits,
            element_is_finite=True,
            round_mode="rn",
        )
        return mxfp_quantizer_sim(
            tensor=x,
            block_dim=block_dim,
            mxfp_meta=meta,
            quantile_search=quantile_search,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None


def mxfp_quantizer(
    x: Tensor,
    block_size: int,
    element_exp_bits: int,
    element_frac_bits: int,
    block_dim: int = -1,
    scale_exp_bits: int = 8,
    quantile_search: bool = False,
) -> Tensor:
    """
    MXFP quantizer with mase-style interface.

    Converts tensor to MXFP format with block-wise shared exponent
    and minifloat elements, then dequantizes back.

    Args:
        x: Input tensor to quantize
        block_size: Number of elements per block for shared exponent (e.g., 32)
        element_exp_bits: Exponent bits for each element (e.g., 4 for E4M3)
        element_frac_bits: Fraction bits for each element (e.g., 3 for E4M3)
        block_dim: Dimension to apply block quantization (-1 for last dim)
        scale_exp_bits: Bits for shared scale exponent (default 8)
        quantile_search: Enable quantile-based clipping search

    Returns:
        Quantized tensor in dequantized form

    Example:
        >>> x = torch.randn(4, 32)
        >>> q = mxfp_quantizer(x, block_size=32, element_exp_bits=4, element_frac_bits=3)

    Common formats:
        - E4M3: element_exp_bits=4, element_frac_bits=3 (8-bit element)
        - E5M2: element_exp_bits=5, element_frac_bits=2 (8-bit element)
    """
    return MXFPQuantize.apply(
        x,
        block_size,
        element_exp_bits,
        element_frac_bits,
        block_dim,
        scale_exp_bits,
        quantile_search,
    )
