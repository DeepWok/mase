"""
MXINT quantizer.
"""

import torch
from torch import Tensor
from tqdm import tqdm

from .meta import MXIntMeta, MXIntTensorMeta
from .fake import extract_mxint_components, compose_mxint_tensor
from ..mxfp.helpers import flatten_for_quantize, permute_for_dequantize


def mxint_quantizer_sim(
    tensor: Tensor,
    block_dim: int,
    mxint_meta: MXIntMeta,
    act_tensor: Tensor | None = None,
    dtype: torch.dtype | None = None,
    quantile_search: bool = False,
    cali_batch_size: int = 32,
) -> Tensor:
    """
    Quantize and dequantize a tensor using MXINT format.

    Args:
        tensor: Input tensor to quantize
        block_dim: Dimension to apply block quantization
        mxint_meta: MXINT format specification
        act_tensor: Optional activation tensor for GPTQ-style calibration
        dtype: Output dtype (default: same as input)
        quantile_search: Enable quantile-based clipping search
        cali_batch_size: Batch size for calibration

    Returns:
        Dequantized tensor
    """
    tensor_dtype = tensor.dtype

    if quantile_search:
        qtensor = tensor.flatten()
        B = mxint_meta.block_size

        qtensor = qtensor.reshape(-1, B)

        percentiles = torch.tensor(
            [1.0, 0.995, 0.99, 0.97, 0.95, 0.93, 0.90, 0.80, 0.70, 0.60, 0.50],
            device=tensor.device,
            dtype=torch.float32,
        )

        device = str(tensor.device)
        ori_shape = tuple(tensor.shape)
        ori_dtype = str(tensor.dtype).removeprefix("torch.")
        ndim = len(ori_shape)
        assert block_dim < ndim and block_dim >= -ndim

        tensor_flat = flatten_for_quantize(tensor, block_dim)

        x = tensor_flat
        n_blocks = x.numel() // B

        x = x.flatten()
        x = x.reshape(n_blocks, B)

        tem_dtype = x.dtype
        x_max = x.abs().to(torch.float32).quantile(percentiles, dim=1, keepdim=True).to(tem_dtype)

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

        quant_tensor = x_mant / 2 ** (mxint_meta.element_bits - 1) * 2 ** (scale - scale_bias)

        del x, x_max, scale, x_mant
        torch.cuda.empty_cache()
        quant_tensor = quant_tensor.reshape(len(percentiles), n_blocks, B)

        if act_tensor is None:
            err = torch.norm(quant_tensor - qtensor, p=2, dim=-1)
            min_err_idx = torch.argmin(err, dim=0).reshape(-1)
        else:
            BATCH_SIZE = cali_batch_size
            last_dim = act_tensor.shape[-1]
            if last_dim != B:
                assert last_dim % B == 0
                act_tensor = act_tensor.view(*act_tensor.shape[:-1], last_dim // B, B)

            total_batches = act_tensor.shape[0]
            err = torch.zeros(
                [percentiles.shape[0], qtensor.shape[0]],
                device=tensor.device,
                dtype=tensor.dtype,
            )

            with torch.no_grad():
                for b in tqdm(range(0, total_batches, BATCH_SIZE), desc="Batching quant output", disable=True):
                    act_b = act_tensor[b:b + BATCH_SIZE]
                    out_orig = torch.matmul(act_b, qtensor.T)
                    out_q = torch.einsum('asb,phb->pash', act_b, quant_tensor.to(act_tensor.dtype))
                    err += torch.norm(out_q - out_orig, p=2, dim=(1, 2))

                    del act_b, out_q, out_orig

            min_err_idx = torch.argmin(err, dim=0)
            torch.cuda.empty_cache()

        quant_tensor = quant_tensor[min_err_idx, torch.arange(quant_tensor.shape[1])]

        tensor_meta = MXIntTensorMeta(
            device=device,
            dtype=ori_dtype,
            shape=ori_shape,
            block_dim=block_dim,
            meta=mxint_meta,
        )

        tensor_out = permute_for_dequantize(
            quant_tensor, ori_shape=tensor_meta.shape, block_dim=tensor_meta.block_dim
        )
        out_dq = tensor_out.to(tensor_dtype)

    else:
        device = str(tensor.device)
        ori_shape = tuple(tensor.shape)
        ori_dtype = str(tensor.dtype).removeprefix("torch.")
        ndim = len(ori_shape)
        assert block_dim < ndim and block_dim >= -ndim

        tensor_flat = flatten_for_quantize(tensor, block_dim)
        scales, elements = extract_mxint_components(tensor_flat, mxint_meta, percentile=1.0)

        tensor_meta = MXIntTensorMeta(
            device=device,
            dtype=ori_dtype,
            shape=ori_shape,
            block_dim=block_dim,
            meta=mxint_meta,
        )

        dequant = compose_mxint_tensor(scales, elements, mxint_meta)
        out_dq = permute_for_dequantize(dequant, tensor_meta.shape, tensor_meta.block_dim)
        out_dq = out_dq.to(dtype or tensor_dtype)

    return out_dq


# =============================================================================
# Mase-style quantizer interface with STE
# =============================================================================


class MXIntQuantize(torch.autograd.Function):
    """Autograd function for MXINT quantization with STE gradient."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        block_size: int,
        element_bits: int,
        block_dim: int,
        scale_bits: int,
        quantile_search: bool,
    ) -> Tensor:
        meta = MXIntMeta(
            block_size=block_size,
            scale_bits=scale_bits,
            element_bits=element_bits,
        )
        return mxint_quantizer_sim(
            tensor=x,
            block_dim=block_dim,
            mxint_meta=meta,
            quantile_search=quantile_search,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None


def mxint_quantizer(
    x: Tensor,
    block_size: int,
    element_bits: int,
    block_dim: int = -1,
    scale_bits: int = 8,
    quantile_search: bool = False,
) -> Tensor:
    """
    MXINT quantizer with mase-style interface.

    Converts tensor to MXINT format with block-wise shared scale
    and integer elements, then dequantizes back.

    Args:
        x: Input tensor to quantize
        block_size: Number of elements per block (e.g., 32)
        element_bits: Bits per element (e.g., 4 or 8)
        block_dim: Dimension to apply block quantization (-1 for last dim)
        scale_bits: Bits for shared scale (default 8)
        quantile_search: Enable quantile-based clipping search

    Returns:
        Quantized tensor in dequantized form

    Example:
        >>> x = torch.randn(4, 32)
        >>> q = mxint_quantizer(x, block_size=32, element_bits=8)

    Common formats:
        - MXINT8: element_bits=8
        - MXINT4: element_bits=4
    """
    return MXIntQuantize.apply(
        x,
        block_size,
        element_bits,
        block_dim,
        scale_bits,
        quantile_search,
    )
