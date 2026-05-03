"""Hadamard rotation utilities (offline + online) for quantized inference.

Ported from Plena-Acc-Sim (which adapted QuaRot / quip-sharp). Provides the
predefined Hadamard base matrices, the hierarchical Hadamard transform, the
fused-into-Linear helper, and a quantizer-style ``mxint_rotate_quantizer``
that applies an exact Hadamard rotation around the MXINT activation quantizer.
"""

from .hadamard_utils import (
    apply_exact_had_to_linear,
    get_hadK,
    is_pow2,
    matmul_hadU,
    matmul_hadUt,
    matmul_hadU_cuda,
    matmul_hadUt_cuda,
    random_hadamard_matrix,
)
from .mxint_rotate import mxint_rotate_quantizer
from .mxfp_rotate import mxfp_rotate_quantizer

__all__ = [
    "apply_exact_had_to_linear",
    "get_hadK",
    "is_pow2",
    "matmul_hadU",
    "matmul_hadUt",
    "matmul_hadU_cuda",
    "matmul_hadUt_cuda",
    "random_hadamard_matrix",
    "mxint_rotate_quantizer",
    "mxfp_rotate_quantizer",
]
