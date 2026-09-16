"""Optional torch.compile fusion for the elementwise rounding primitives.

The fake-quantised decode step is launch-bound: every vector rounding, matrix
writeout and MX quantiser call is a chain of 10-30 tiny elementwise kernels.
Compiling those pure functions fuses each chain into one or two kernels. The
compiled outputs are expected to be bit-identical to eager; the helpers here
never change numerics, only how the same ops are launched.

Fusion is opt-in. Set ``MASE_COMPILE_ROUNDING=1`` in the environment before
``chop`` is imported to enable it; with the variable unset (or any other
value) every helper returns the eager function unchanged and nothing in
``torch._dynamo`` is touched.

Format parameters (bit widths, block sizes, rounding modes) must be
compile-time constants because the eager code shifts by them, while tensor
shapes must stay dynamic: autoregressive decode changes the sequence length
every step, and a global ``specialize_int`` caused a recompile per generated
token. ``compiled_variant`` therefore compiles one closure per distinct
constant tuple and leaves dynamo's shape handling at its default.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("MASE_COMPILE_ROUNDING", "0") == "1"

_VARIANTS: dict[tuple, object] = {}
_DYNAMO_CONFIGURED = False


def _configure_dynamo() -> None:
    """Raise dynamo's cache limits and enable eager fallback, once.

    Called lazily from the first real ``torch.compile`` so that merely
    importing this module has no side effect on global torch state.
    """

    global _DYNAMO_CONFIGURED
    if _DYNAMO_CONFIGURED:
        return
    import torch._dynamo

    config = torch._dynamo.config
    config.cache_size_limit = max(int(getattr(config, "cache_size_limit", 8)), 128)
    config.accumulated_cache_size_limit = max(
        int(getattr(config, "accumulated_cache_size_limit", 256)), 4096
    )
    # Any compile failure falls back to the eager implementation.
    config.suppress_errors = True
    _DYNAMO_CONFIGURED = True


def _compile(function):
    _configure_dynamo()
    return torch.compile(function, dynamic=None)


def maybe_compile(function):
    """Return a fused version of ``function`` when compilation is enabled."""

    if not ENABLED:
        return function
    return _compile(function)


def compiled_variant(key: tuple, builder):
    """Return (and cache) a compiled closure for one constant tuple.

    ``builder()`` must return a function of tensors only, with every format
    constant captured in its closure, so dynamo bakes the constants in and
    never turns them into symbolic ints. When compilation is disabled the
    builder's function is returned as-is.
    """

    variant = _VARIANTS.get(key)
    if variant is None:
        function = builder()
        variant = _compile(function) if ENABLED else function
        _VARIANTS[key] = variant
    return variant


__all__ = ["ENABLED", "maybe_compile", "compiled_variant"]
