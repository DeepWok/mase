"""Online Hadamard rotation around the MXFP activation quantizer.

Quantizer-style API parallel to ``mxfp_quantizer``: takes a tensor in,
returns the quantized-then-derotated tensor out. Hadamard base matrices are
cached per (size, device, dtype) by ``mxint_rotate``'s ``_hadamard_factors``.
"""

import warnings

import torch
from torch import Tensor

from chop.nn.quantizers.mxfp import mxfp_quantizer
from .hadamard_utils import matmul_hadU, matmul_hadU_cuda
from .mxint_rotate import _hadamard_factors


def mxfp_rotate_quantizer(
    x: Tensor,
    hadamard_dim: int,
    block_size: int,
    element_exp_bits: int,
    element_frac_bits: int,
    block_dim: int = -1,
    scale_exp_bits: int = 8,
    quantile_search: bool = False,
    force_fp32: bool = False,
) -> Tensor:
    """Apply exact Hadamard rotation around an MXFP activation quantizer.

        H @ x  ->  mxfp_quant  ->  H.T @ x

    Mathematically a no-op in fp (the two rotations cancel), but redistributes
    outliers inside each MXFP block, reducing quantization error on
    activations with channel-wise spikes (QuaRot online rotation).

    For the rotation pair to cancel correctly under the surrounding linears,
    the upstream weight matrices must be offline-rotated to match — typically
    used on layers whose inputs feed o_proj / down_proj after running the
    rotate pass with ``online_rotate=True``.

    Args:
        x: Input tensor; rotation/quantization happens along the last dim.
        hadamard_dim: Size of the rotation, normally ``in_features``.
        block_size, element_exp_bits, element_frac_bits, block_dim,
            scale_exp_bits, quantile_search: Forwarded to ``mxfp_quantizer``.
        force_fp32: Run the Hadamard multiplications in fp32 for numerical
            headroom before casting back to the input dtype.
    """
    x_dtype = x.dtype
    had_dtype = torch.float32 if force_fp32 else x_dtype
    dtype_str = str(had_dtype).removeprefix("torch.")

    factors = _hadamard_factors(int(hadamard_dim), str(x.device), dtype_str)
    if factors is None:
        warnings.warn(
            f"mxfp_rotate_quantizer: dimension {hadamard_dim} is not supported "
            "by the Hadamard decomposition table; falling back to plain MXFP.",
            RuntimeWarning,
        )
        return mxfp_quantizer(
            x,
            block_size=block_size,
            element_exp_bits=element_exp_bits,
            element_frac_bits=element_frac_bits,
            block_dim=block_dim,
            scale_exp_bits=scale_exp_bits,
            quantile_search=quantile_search,
        )
    had_K, K, had_K_T, K_T = factors

    if force_fp32:
        x = x.float()

    if x.is_cuda:
        x = matmul_hadU_cuda(x, had_K, K)
    else:
        x = matmul_hadU(x)

    x = mxfp_quantizer(
        x,
        block_size=block_size,
        element_exp_bits=element_exp_bits,
        element_frac_bits=element_frac_bits,
        block_dim=block_dim,
        scale_exp_bits=scale_exp_bits,
        quantile_search=quantile_search,
    )

    if x.is_cuda:
        x = matmul_hadU_cuda(x, had_K_T, K_T)
    else:
        x = matmul_hadU(x, transpose=True)

    return x.to(x_dtype)
