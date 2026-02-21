"""
Minifloat metadata for MX-format quantizers.
"""

import functools
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MinifloatMeta:
    """
    Metadata for minifloat types.

    Args:
        exp_bits: Number of exponent bits
        frac_bits: Number of fraction bits
        is_finite: Whether the minifloat type supports inf/nan
        round_mode: Rounding mode - "rn" (nearest), "rd" (down), "ru" (up), "rz" (truncate)
    """

    exp_bits: int
    frac_bits: int
    is_finite: bool
    round_mode: Literal["rn", "rd", "ru", "rz"]

    def __post_init__(self):
        assert self.exp_bits > 0
        assert self.frac_bits > 0
        assert self.exp_bits + self.frac_bits < 16

    @functools.cached_property
    def n_bits(self) -> int:
        return self.exp_bits + self.frac_bits + 1


@dataclass
class MinifloatTensorMeta:
    device: str
    dtype: str
    shape: tuple[int, ...]
    meta: MinifloatMeta
