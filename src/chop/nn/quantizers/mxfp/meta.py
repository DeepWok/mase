"""
MXFP metadata classes.
"""

import functools
from dataclasses import dataclass
from typing import Literal

from .._minifloat_mx import MinifloatMeta


@dataclass(frozen=True)
class MXFPMeta:
    """
    Metadata for MXFP (Mixed-exponent Floating Point) format.

    MXFP uses block-wise shared exponents with minifloat elements.

    Args:
        block_size: Number of elements per block for shared exponent
        scale_exp_bits: Bits for shared scale exponent (typically 8)
        element_exp_bits: Exponent bits per element (e.g., 4 for E4M3)
        element_frac_bits: Fraction bits per element (e.g., 3 for E4M3)
        element_is_finite: Whether elements support inf/nan
        round_mode: Rounding mode
    """

    block_size: int
    scale_exp_bits: int
    element_exp_bits: int
    element_frac_bits: int
    element_is_finite: bool
    round_mode: Literal["rn", "ru", "rd", "rz"]

    def __post_init__(self):
        legal_scale_exp_bits = (8,)
        assert self.scale_exp_bits in legal_scale_exp_bits, (
            f"Invalid scale exponent bits: {self.scale_exp_bits}. "
            f"Legal values are: {legal_scale_exp_bits}."
        )
        legal_element_exp_frac_bits = ((4, 3), (5, 2), (2, 3), (3, 2), (2, 1), (1, 2))
        el_exp_frac = (self.element_exp_bits, self.element_frac_bits)
        assert el_exp_frac in legal_element_exp_frac_bits, (
            f"Invalid element exp/frac bits: {el_exp_frac}. "
            f"Legal values are: {legal_element_exp_frac_bits}."
        )

    @functools.cached_property
    def element_meta(self) -> MinifloatMeta:
        """Returns MinifloatMeta for the element part of MXFP format."""
        return MinifloatMeta(
            exp_bits=self.element_exp_bits,
            frac_bits=self.element_frac_bits,
            is_finite=self.element_is_finite,
            round_mode=self.round_mode,
        )


@dataclass(frozen=True)
class MXFPTensorMeta:
    """Runtime metadata for an MXFP tensor."""

    device: str
    dtype: str
    shape: tuple[int, ...]
    block_dim: int
    meta: MXFPMeta
