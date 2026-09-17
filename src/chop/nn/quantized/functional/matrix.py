"""Numerical model of the PLENA matrix-unit output boundary.

The matrix unit reduces each MLEN-wide instruction partition into a signed
16.16 fixed-point bank (``fp_fix_accumulator``), accumulates partitions in that
bank, then truncates the result to the vector datapath format on writeout.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .vector import VectorRoundingPolicy, parse_vector_format
from chop.nn.quantizers._compile import maybe_compile

# Signed fixed-point accumulator geometry of the MX systolic PE.
ACCUMULATOR_INTEGER_BITS = 16
ACCUMULATOR_FRACTION_BITS = 16


def _truncate_to_accumulator(tensor: Tensor) -> Tensor:
    """Truncate toward zero into the signed 16.16 bank, wrapping on overflow."""
    work = tensor.to(torch.float32)
    finite = torch.isfinite(work)
    scale = 2.0**ACCUMULATOR_FRACTION_BITS
    total_bits = ACCUMULATOR_INTEGER_BITS + ACCUMULATOR_FRACTION_BITS
    fixed = torch.trunc(torch.where(finite, work, torch.zeros_like(work)) * scale)
    fixed = fixed.to(torch.int64)
    fixed = torch.remainder(fixed + 2 ** (total_bits - 1), 2**total_bits) - 2 ** (
        total_bits - 1
    )
    return torch.where(finite, fixed.to(torch.float32) / scale, work)


def _truncate_to_vector_format(tensor: Tensor, token: str) -> Tensor:
    """Truncate magnitude to the representable grid of one vector format."""
    vector_format = parse_vector_format(token)
    work = tensor.to(torch.float32)
    if vector_format.is_bfloat16:
        bits = work.contiguous().view(torch.int32)
        return (bits & -65536).view(torch.float32)

    magnitude = work.abs()
    finite = torch.isfinite(magnitude)
    nonzero_finite = finite & (magnitude != 0)
    safe = torch.where(nonzero_finite, magnitude, torch.ones_like(magnitude))
    bias = (1 << (vector_format.exponent_bits - 1)) - 1
    min_exponent = 1 - bias
    max_exponent = (1 << vector_format.exponent_bits) - 2 - bias
    exponent = torch.floor(torch.log2(safe)).clamp(
        min=min_exponent,
        max=max_exponent,
    )
    normal_step = torch.exp2(exponent - vector_format.fraction_bits)
    subnormal_step = 2.0 ** (min_exponent - vector_format.fraction_bits)
    step = torch.where(magnitude < 2.0**min_exponent, subnormal_step, normal_step)
    truncated = torch.floor(magnitude / step) * step
    max_finite = (2.0 - 2.0 ** (-vector_format.fraction_bits)) * 2.0**max_exponent
    truncated = truncated.clamp(max=max_finite)
    truncated = torch.where(nonzero_finite, truncated, magnitude)
    truncated = torch.where(finite, truncated, work)
    return torch.copysign(truncated, work)


truncate_to_accumulator = maybe_compile(_truncate_to_accumulator)
truncate_to_vector_format = maybe_compile(_truncate_to_vector_format)


def _matrix_config(config: dict | None) -> tuple[str, int] | None:
    """Return (output_format, matrix_mlen), or None when unbound."""
    cfg = config or {}
    output_format = cfg.get("output_format")
    matrix_mlen = cfg.get("matrix_mlen")
    if output_format is None and matrix_mlen is None:
        return None
    if not isinstance(output_format, str):
        raise ValueError("matrix output requires a canonical output_format token")
    if (
        isinstance(matrix_mlen, bool)
        or not isinstance(matrix_mlen, int)
        or matrix_mlen <= 0
    ):
        raise ValueError("matrix_mlen must be a positive integer")
    return output_format, matrix_mlen


def plena_matrix_output(tensor: Tensor, config: dict | None) -> Tensor:
    """Apply the fixed-bank and vector-writeout boundaries to a matrix result."""
    bound = _matrix_config(config)
    if bound is None:
        return tensor
    output_format, _ = bound
    accumulator = truncate_to_accumulator(tensor)
    return truncate_to_vector_format(accumulator, output_format).to(tensor.dtype)


def plena_matrix_product(lhs: Tensor, rhs: Tensor, config: dict | None) -> Tensor:
    """Evaluate one matrix product with per-instruction partition boundaries."""
    bound = _matrix_config(config)
    if bound is None:
        return torch.matmul(lhs, rhs)
    output_format, matrix_mlen = bound
    if lhs.shape[-1] != rhs.shape[-2]:
        raise ValueError("matrix operands have incompatible reduction dimensions")
    storage = VectorRoundingPolicy.from_token(output_format)
    accumulator = None
    for start in range(0, lhs.shape[-1], matrix_mlen):
        stop = min(lhs.shape[-1], start + matrix_mlen)
        partial = torch.matmul(
            lhs[..., start:stop].to(torch.float32),
            rhs[..., start:stop, :].to(torch.float32),
        )
        fixed_partial = truncate_to_accumulator(storage.round(partial))
        accumulator = (
            fixed_partial
            if accumulator is None
            else truncate_to_accumulator(accumulator + fixed_partial)
        )
    return truncate_to_vector_format(accumulator, output_format).to(lhs.dtype)


def scale_matrix_output(
    tensor: Tensor,
    scaling: float,
    config: dict | None,
) -> Tensor:
    """Round a post-matmul scale at the vector writeout boundary."""
    bound = _matrix_config(config)
    if bound is None:
        return tensor * scaling
    output_format, _ = bound
    scaled = tensor.to(torch.float32) * scaling
    return truncate_to_vector_format(scaled, output_format).to(tensor.dtype)


__all__ = [
    "ACCUMULATOR_FRACTION_BITS",
    "ACCUMULATOR_INTEGER_BITS",
    "plena_matrix_output",
    "plena_matrix_product",
    "scale_matrix_output",
    "truncate_to_accumulator",
    "truncate_to_vector_format",
]
