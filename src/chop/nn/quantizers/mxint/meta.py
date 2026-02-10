"""
MXINT metadata classes.
"""

from dataclasses import dataclass


@dataclass
class MXIntMeta:
    """
    Metadata for MXINT (Mixed-exponent Integer) format.

    MXINT uses block-wise shared scale with integer elements.

    Args:
        block_size: Number of elements per block
        scale_bits: Bits for shared scale (typically 8)
        element_bits: Bits per element (e.g., 4 or 8)
    """

    block_size: int
    scale_bits: int
    element_bits: int


@dataclass
class MXIntTensorMeta:
    """Runtime metadata for an MXINT tensor."""

    device: str
    dtype: str
    shape: tuple[int, ...]
    block_dim: int
    meta: MXIntMeta
