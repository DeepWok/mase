"""Optional torch.compile fusion for the elementwise rounding primitives.

The fake-quantised decode step is launch-bound: every vector-rounding, matrix
writeout and MX quantiser call is a chain of 10-30 tiny elementwise kernels.
Compiling those pure functions fuses each chain into one or two kernels; the
compiled outputs are bit-identical to eager (verified by
``PLENA_Qwen30B_Moe_Results/scripts/test_compiled_rounding.py``).

Format parameters (bit widths, block sizes, rounding modes) must be compile-time
constants — the eager code shifts by them — while tensor shapes must stay
dynamic (autoregressive decode changes the sequence length every step, and a
global ``specialize_int`` caused a recompile per generated token). The pattern
here therefore compiles one variant per distinct constant tuple via
``compiled_variant`` and leaves the dynamo shape handling at its default.

Set ``MASE_COMPILE_ROUNDING=0`` to disable everything.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("MASE_COMPILE_ROUNDING", "1") != "0"

if ENABLED:
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = max(
        int(getattr(torch._dynamo.config, "cache_size_limit", 8)), 128
    )
    torch._dynamo.config.accumulated_cache_size_limit = max(
        int(getattr(torch._dynamo.config, "accumulated_cache_size_limit", 256)),
        4096,
    )
    # Any compile failure falls back to the eager implementation.
    torch._dynamo.config.suppress_errors = True

_VARIANTS: dict[tuple, object] = {}


def maybe_compile(function):
    """Return a fused version of ``function`` when compilation is enabled."""

    if not ENABLED:
        return function
    return torch.compile(function, dynamic=None)


def compiled_variant(key: tuple, builder):
    """Return (and cache) a compiled closure for one constant tuple.

    ``builder()`` must return a function of tensors only, with every format
    constant captured in its closure, so dynamo bakes the constants in and
    never turns them into symbolic ints.
    """

    variant = _VARIANTS.get(key)
    if variant is None:
        function = builder()
        variant = torch.compile(function, dynamic=None) if ENABLED else function
        _VARIANTS[key] = variant
    return variant


__all__ = ["ENABLED", "maybe_compile", "compiled_variant"]
