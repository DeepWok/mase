"""The optional torch.compile fusion of the rounding primitives.

``MASE_COMPILE_ROUNDING`` is read once when ``chop.nn.quantizers._compile`` is
imported, so each setting is exercised in a fresh interpreter. The child
process quantises fixed tensors with every fused entry point and writes the
results out; the parent recomputes them eagerly and requires bit equality.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

_WORKLOAD = textwrap.dedent(
    """
    import sys
    import torch
    import torch._dynamo

    before = (
        torch._dynamo.config.cache_size_limit,
        torch._dynamo.config.accumulated_cache_size_limit,
        torch._dynamo.config.suppress_errors,
    )

    from chop.nn.quantizers import _compile
    from chop.nn.quantizers._minifloat_mx import MinifloatMeta
    from chop.nn.quantizers._minifloat_mx.fake import quantize_minifloat_value
    from chop.nn.quantizers.mxfp import mxfp_quantizer
    from chop.nn.quantizers.mxint import mxint_quantizer
    from chop.nn.quantized.functional.matrix import (
        truncate_to_accumulator,
        truncate_to_vector_format,
    )

    after_import = (
        torch._dynamo.config.cache_size_limit,
        torch._dynamo.config.accumulated_cache_size_limit,
        torch._dynamo.config.suppress_errors,
    )

    torch.manual_seed(0)
    x = torch.randn(4, 64) * 8
    meta = MinifloatMeta(exp_bits=4, frac_bits=3, is_finite=True, round_mode="rn")
    out = {
        "enabled": _compile.ENABLED,
        "dynamo_before": before,
        "dynamo_after_import": after_import,
        "mxfp": mxfp_quantizer(x, 32, 4, 3, -1, 8),
        "mxint": mxint_quantizer(x, 32, 8, -1, 8),
        "minifloat": quantize_minifloat_value(x, meta),
        "accumulator": truncate_to_accumulator(x),
        "vector": truncate_to_vector_format(x, "FP_E3M2"),
    }
    out["variants"] = sorted(_compile._VARIANTS)
    torch.save(out, sys.argv[1])
    """
)


def _run_child(tmp_path, env_value):
    env = dict(os.environ)
    env.pop("MASE_COMPILE_ROUNDING", None)
    if env_value is not None:
        env["MASE_COMPILE_ROUNDING"] = env_value
    target = tmp_path / f"out-{env_value}.pt"
    subprocess.run(
        [sys.executable, "-c", _WORKLOAD, str(target)],
        env=env,
        check=True,
        timeout=600,
    )
    return torch.load(target, weights_only=False)


def _eager_reference():
    from chop.nn.quantizers._minifloat_mx import MinifloatMeta
    from chop.nn.quantizers._minifloat_mx.fake import _quantize_minifloat_value
    from chop.nn.quantizers.mxfp.mxfp import MXFPQuantize
    from chop.nn.quantizers.mxint.mxint import MXIntQuantize
    from chop.nn.quantized.functional.matrix import (
        _truncate_to_accumulator,
        _truncate_to_vector_format,
    )

    torch.manual_seed(0)
    x = torch.randn(4, 64) * 8
    meta = MinifloatMeta(exp_bits=4, frac_bits=3, is_finite=True, round_mode="rn")
    return {
        "mxfp": MXFPQuantize.apply(x, 32, 4, 3, -1, 8, False),
        "mxint": MXIntQuantize.apply(x, 32, 8, -1, 8, False),
        "minifloat": _quantize_minifloat_value(x, meta),
        "accumulator": _truncate_to_accumulator(x),
        "vector": _truncate_to_vector_format(x, "FP_E3M2"),
    }


_KEYS = ("mxfp", "mxint", "minifloat", "accumulator", "vector")


@pytest.mark.parametrize("env_value", [None, "0", "yes"])
def test_rounding_is_eager_unless_opted_in(tmp_path, env_value):
    result = _run_child(tmp_path, env_value)
    assert result["enabled"] is False
    # Importing chop must not mutate global dynamo configuration.
    assert result["dynamo_after_import"] == result["dynamo_before"]
    assert result["variants"] == []
    reference = _eager_reference()
    for key in _KEYS:
        assert torch.equal(result[key], reference[key]), key


def test_env_var_enables_compiled_variants(tmp_path):
    result = _run_child(tmp_path, "1")
    assert result["enabled"] is True
    # Opting in is what raises the dynamo cache limits and enables fallback.
    cache_limit, accumulated_limit, suppress_errors = result["dynamo_after_import"]
    assert cache_limit >= 128 and accumulated_limit >= 4096 and suppress_errors
    # One cached variant per distinct constant tuple that was exercised (the
    # MXFP path also registers the minifloat variant for its element rounding).
    assert {k[0] for k in result["variants"]} == {"minifloat", "mxfp", "mxint"}
    reference = _eager_reference()
    for key in _KEYS:
        assert torch.equal(result[key], reference[key]), key
