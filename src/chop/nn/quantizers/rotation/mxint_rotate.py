"""Online Hadamard rotation around the MXINT activation quantizer.

Quantizer-style API parallel to ``mxint_quantizer``: takes a tensor in,
returns the quantized-then-derotated tensor out. Hadamard base matrices are
cached per (size, device, dtype) so they're materialised once.
"""

from functools import lru_cache
import warnings

import torch
from torch import Tensor

from chop.nn.quantizers.mxint import mxint_quantizer
from .hadamard_utils import get_hadK, matmul_hadU, matmul_hadU_cuda


@lru_cache(maxsize=64)
def _hadamard_factors(hadamard_dim: int, device: str, dtype_str: str):
    """Return cached (had_K, K, had_K_T, K_T) for the requested dimension.

    Returns None when ``hadamard_dim`` cannot be decomposed by ``get_hadK``.
    ``had_K`` itself may be None when ``K == 1`` (pure FHT path).
    """
    try:
        had_K, K = get_hadK(hadamard_dim)
        had_K_T, K_T = get_hadK(hadamard_dim, transpose=True)
    except (AssertionError, ValueError):
        return None

    dtype = getattr(torch, dtype_str)
    if had_K is not None:
        had_K = had_K.to(device=device, dtype=dtype)
        had_K_T = had_K_T.to(device=device, dtype=dtype)
    return had_K, K, had_K_T, K_T


def mxint_rotate_quantizer(
    x: Tensor,
    hadamard_dim: int,
    block_size: int,
    element_bits: int,
    block_dim: int = -1,
    scale_bits: int = 8,
    quantile_search: bool = False,
    force_fp32: bool = False,
) -> Tensor:
    """Apply exact Hadamard rotation around an MXINT activation quantizer.

        H @ x  ->  mxint_quant  ->  H.T @ x

    Mathematically a no-op in fp (the two rotations cancel), but redistributes
    outliers inside each MXINT block, reducing quantization error on
    activations with channel-wise spikes (QuaRot online rotation).

    For the rotation pair to cancel correctly under the surrounding linears,
    the upstream weight matrices must be offline-rotated to match — typically
    used on layers whose inputs feed o_proj / down_proj after running the
    rotate pass with ``online_rotate=True``.

    Args:
        x: Input tensor; rotation/quantization happens along the last dim.
        hadamard_dim: Size of the rotation, normally ``in_features``.
        block_size, element_bits, block_dim, scale_bits, quantile_search:
            Forwarded to ``mxint_quantizer``.
        force_fp32: Run the Hadamard multiplications in fp32 for numerical
            headroom before casting back to the input dtype.
    """
    x_dtype = x.dtype
    had_dtype = torch.float32 if force_fp32 else x_dtype
    dtype_str = str(had_dtype).removeprefix("torch.")

    factors = _hadamard_factors(int(hadamard_dim), str(x.device), dtype_str)
    if factors is None:
        warnings.warn(
            f"mxint_rotate_quantizer: dimension {hadamard_dim} is not supported "
            "by the Hadamard decomposition table; falling back to plain MXINT.",
            RuntimeWarning,
        )
        return mxint_quantizer(
            x,
            block_size=block_size,
            element_bits=element_bits,
            block_dim=block_dim,
            scale_bits=scale_bits,
            quantile_search=quantile_search,
        )
    had_K, K, had_K_T, K_T = factors

    if force_fp32:
        x = x.float()

    # CUDA path uses fast_hadamard_transform (lazy-required); CPU path uses
    # the pure-torch decomposition for local debugging / smoke tests.
    if x.is_cuda:
        x = matmul_hadU_cuda(x, had_K, K)
    else:
        x = matmul_hadU(x)

    x = mxint_quantizer(
        x,
        block_size=block_size,
        element_bits=element_bits,
        block_dim=block_dim,
        scale_bits=scale_bits,
        quantile_search=quantile_search,
    )

    if x.is_cuda:
        x = matmul_hadU_cuda(x, had_K_T, K_T)
    else:
        x = matmul_hadU(x, transpose=True)

    return x.to(x_dtype)
