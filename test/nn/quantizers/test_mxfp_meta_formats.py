"""MXFPMeta accepts every supported element format and rejects the rest."""

import pytest
import torch

from chop.nn.quantizers.mxfp import mxfp_quantizer_sim
from chop.nn.quantizers.mxfp.meta import MXFPMeta


def _meta(exp_bits: int, frac_bits: int) -> MXFPMeta:
    return MXFPMeta(
        block_size=32,
        scale_exp_bits=8,
        element_exp_bits=exp_bits,
        element_frac_bits=frac_bits,
        element_is_finite=True,
        round_mode="rn",
    )


@pytest.mark.parametrize(
    "exp_bits,frac_bits",
    [(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (5, 2)],
    ids=["E1M2", "E2M1", "E2M3", "E3M2", "E3M4", "E4M3", "E5M2"],
)
def test_legal_element_formats_construct_and_quantise(exp_bits, frac_bits):
    meta = _meta(exp_bits, frac_bits)
    x = torch.randn(4, 64)
    q = mxfp_quantizer_sim(x, block_dim=-1, mxfp_meta=meta)
    assert q.shape == x.shape
    assert torch.isfinite(q).all()


def test_e3m4_is_an_eight_bit_format():
    meta = _meta(3, 4)
    assert meta.element_exp_bits + meta.element_frac_bits + 1 == 8


@pytest.mark.parametrize("exp_bits,frac_bits", [(3, 5), (6, 1), (1, 1)])
def test_unsupported_element_formats_are_rejected(exp_bits, frac_bits):
    with pytest.raises(AssertionError):
        _meta(exp_bits, frac_bits)
