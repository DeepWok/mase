"""Shared runtime phase context for quantized decoder-only inference.

Phase-split quantisation targets disaggregated serving (e.g. PLENA), where
prefill and decode run on separate accelerators that can carry different
quantisation profiles. This module centralizes the runtime phase state
(``prefill`` vs ``decode``) in one place so quantized modules stay loosely
coupled:

1. Decoder-layer pre-hooks (installed by the quantize pass) detect the
   runtime phase from cache semantics before the layer body executes.
2. Downstream quantized modules (attention / linear / mlp / rms_norm) read
   the same phase without changing their public ``forward(...)`` signatures.

Why ``ContextVar`` is used:
- It avoids global mutable state bleeding across threads/tasks.
- It keeps the integration minimally invasive for the existing MASE module
  API.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

Phase = Literal["prefill", "decode"]
DecodePolicy = Literal["fp_only", "quantized"]


_ACTIVE_PHASE: ContextVar[Phase] = ContextVar("active_quant_phase", default="prefill")
# Process-wide (not a ContextVar): an evaluation-time override must also be
# visible to forwards running in worker threads (e.g. nn.DataParallel), which
# do not inherit the caller's ContextVar state.
_PHASE_OVERRIDE: Phase | None = None
_DECODE_POLICY: ContextVar[DecodePolicy] = ContextVar(
    "active_decode_policy", default="fp_only"
)


def set_runtime_phase(phase: Phase) -> None:
    """Set runtime phase used by phase-aware quantized modules."""

    if phase not in ("prefill", "decode"):
        raise ValueError(f"Unsupported phase: {phase}")
    _ACTIVE_PHASE.set(phase)


def get_runtime_phase() -> Phase:
    """Return current runtime phase.

    An active :func:`force_runtime_phase` override wins over the phase
    written by decoder-layer hooks. Defaults to ``prefill`` when nothing has
    been written yet, so a plain single-shot forward behaves like prefill.
    """

    if _PHASE_OVERRIDE is not None:
        return _PHASE_OVERRIDE
    return _ACTIVE_PHASE.get()


@contextmanager
def force_runtime_phase(phase: Phase):
    """Force all phase-aware modules to a fixed phase within the block.

    The override takes precedence over the decoder-layer pre-hooks (which
    keep inferring the phase from cache semantics but cannot win while the
    override is active). It is process-wide, so it also reaches forwards
    running in worker threads. Primary use case: evaluating decode-chip
    numerics with full-sequence forwards — e.g. WikiText perplexity of the
    quantised decode configuration, or rotation-search scoring — where every
    token should flow through the decode-side quantisation even though the
    forward itself looks like a prefill::

        with force_runtime_phase("decode"):
            ppl = evaluate_perplexity(model, loader)
    """

    global _PHASE_OVERRIDE
    if phase not in ("prefill", "decode"):
        raise ValueError(f"Unsupported phase: {phase}")
    previous = _PHASE_OVERRIDE
    _PHASE_OVERRIDE = phase
    try:
        yield
    finally:
        _PHASE_OVERRIDE = previous


def set_runtime_decode_policy(policy: DecodePolicy) -> None:
    """Set decode policy in runtime phase context.

    Supported policies:
    - ``fp_only``: decode path is forced to full precision.
    - ``quantized``: decode path consumes decode-phase quantized configs.

    The authoritative per-module policy lives in each module's normalized
    phase config; this context mirror exists so external tooling (e.g. the
    PLENA software stack) can introspect the active policy of a forward call.
    """

    if policy not in ("fp_only", "quantized"):
        raise ValueError(f"Unsupported decode policy: {policy}")
    _DECODE_POLICY.set(policy)


def get_runtime_decode_policy() -> DecodePolicy:
    """Return decode policy from runtime phase context."""

    return _DECODE_POLICY.get()


def infer_runtime_phase_from_hidden_and_cache(
    hidden_states: Any, past_cache: Any, cache_position: Any = None
) -> Phase:
    """Infer runtime phase from the current forward's inputs.

    Rules, in order:
    1) ``past_cache is None`` -> ``prefill``
    2) multi-token forward (query length > 1) -> ``prefill``
    3) ``cache_position`` starting at 0 -> ``prefill`` (single-token prompt)
    4) non-empty cache -> ``decode``; empty cache -> ``prefill``

    Rule 2 is what makes the per-layer hooks consistent: during a cached
    prefill, layer 0's attention writes the prompt's KV into the cache
    before layer 1's hook fires, so cache length alone would misclassify
    every later layer as decode. The query length is the same for all
    layers of one forward, so it classifies them identically.

    The same heuristic must be shared by all decoder layers, so it lives in
    this shared module rather than inside any single attention class.
    """

    if past_cache is None:
        return "prefill"

    q_len = 0
    shape = getattr(hidden_states, "shape", None)
    if shape is not None and len(shape) >= 2:
        q_len = int(shape[-2])
    if q_len > 1:
        return "prefill"

    if cache_position is not None:
        try:
            if len(cache_position) > 0:
                return "prefill" if int(cache_position[0]) == 0 else "decode"
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    past_len = 0
    get_seq_length = getattr(past_cache, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            past_len = int(get_seq_length())
        except Exception:  # pragma: no cover - defensive for custom Cache impls
            past_len = 0

    return "decode" if past_len > 0 else "prefill"


def extract_decoder_layer_past_cache(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Extract past-cache object from decoder-layer call arguments.

    Compatibility:
    - New HF naming: ``past_key_values``
    - Legacy alias: ``past_key_value``
    - Positional call fallback: 4th argument of ``LlamaDecoderLayer.forward``
      (same slot for Qwen3).
    """

    if "past_key_values" in kwargs:
        return kwargs["past_key_values"]
    if "past_key_value" in kwargs:
        return kwargs["past_key_value"]
    if len(args) >= 4:
        return args[3]
    return None


def infer_runtime_phase_from_decoder_layer_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Phase:
    """Infer runtime phase from decoder-layer forward inputs."""

    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    past_cache = extract_decoder_layer_past_cache(args, kwargs)
    cache_position = kwargs.get("cache_position")
    return infer_runtime_phase_from_hidden_and_cache(
        hidden_states, past_cache, cache_position
    )
