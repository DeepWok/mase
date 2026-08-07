"""Regression tests for MX block quantization correctness."""

import torch

from chop.nn.quantizers.mxfp.mxfp import mxfp_quantizer
from chop.nn.quantizers.mxint.mxint import mxint_quantizer

MXFP_FORMATS = [(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (5, 2)]
MXINT_WIDTHS = [2, 4, 8]


def test_blocks_do_not_straddle_rows():
    """A block axis that is not a multiple of the block size must still
    keep each row independent: perturbing one row cannot change another."""
    torch.manual_seed(0)
    block_size, rows, axis = 32, 4, 80  # 80 % 32 != 0
    x = torch.randn(rows, axis)
    kwargs = dict(
        block_size=block_size,
        element_exp_bits=4,
        element_frac_bits=3,
        block_dim=-1,
    )
    baseline = mxfp_quantizer(x, **kwargs)
    perturbed = x.clone()
    perturbed[0] *= 100.0
    result = mxfp_quantizer(perturbed, **kwargs)
    for row in range(1, rows):
        assert torch.equal(
            baseline[row], result[row]
        ), f"row {row} changed when only row 0 was perturbed"


def test_mxfp_requantization_is_idempotent():
    """Quantizing an already-quantized tensor must be a no-op, because a
    cached value is re-read many times after one quantization round trip."""
    for exp_bits, frac_bits in MXFP_FORMATS:
        torch.manual_seed(0)
        x = torch.randn(256, 512)
        kwargs = dict(
            block_size=32,
            element_exp_bits=exp_bits,
            element_frac_bits=frac_bits,
            block_dim=-1,
        )
        once = mxfp_quantizer(x, **kwargs)
        twice = mxfp_quantizer(once, **kwargs)
        assert torch.equal(once, twice), f"E{exp_bits}M{frac_bits} not idempotent"


def test_mxint_requantization_is_idempotent():
    for element_bits in MXINT_WIDTHS:
        torch.manual_seed(0)
        x = torch.randn(256, 512)
        kwargs = dict(block_size=32, element_bits=element_bits, block_dim=-1)
        once = mxint_quantizer(x, **kwargs)
        twice = mxint_quantizer(once, **kwargs)
        assert torch.equal(once, twice), f"MXINT{element_bits} not idempotent"


def test_shape_and_dtype_are_preserved():
    torch.manual_seed(0)
    for shape, block_dim in [((8, 256), -1), ((80, 4), 0), ((4, 80), -1)]:
        x = torch.randn(*shape)
        out = mxfp_quantizer(
            x,
            block_size=32,
            element_exp_bits=4,
            element_frac_bits=3,
            block_dim=block_dim,
        )
        assert out.shape == x.shape
        assert out.dtype == x.dtype


def test_small_block_maxima_are_not_flushed_to_zero():
    """A block whose maximum is small but nonzero must not be zeroed,
    including at half precision."""
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        torch.manual_seed(0)
        x = (torch.randn(4, 64) * 1e-7).to(dtype)
        assert x.abs().amax() > 0
        out = mxint_quantizer(x, block_size=32, element_bits=4, block_dim=-1)
        assert not bool((out == 0).all()), f"{dtype} block flushed to zero"
