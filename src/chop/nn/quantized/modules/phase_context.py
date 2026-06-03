"""Shared runtime phase context for quantized decoder-only inference.

This module intentionally centralizes phase state (`prefill` vs `decode`) in a
single place so quantized modules can stay loosely coupled:

1. Llama decoder-layer pre-hooks detect runtime phase from cache semantics.
2. Downstream quantized modules (attention/linear/mlp/rms) read the same phase
   without changing
   their public `forward(...)` signatures.

Why `ContextVar` is used:
- It avoids global mutable state bleeding across threads/tasks.
- It keeps the integration minimally invasive for the existing MASE module API.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Literal

Phase = Literal["prefill", "decode"]
DecodePolicy = Literal["fp_only", "quantized"]


_ACTIVE_PHASE: ContextVar[Phase] = ContextVar("active_quant_phase", default="prefill")
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

    Defaults to `prefill` when no explicit phase has been written yet.
    """

    return _ACTIVE_PHASE.get()


def set_runtime_decode_policy(policy: DecodePolicy) -> None:
    """Set decode policy in runtime phase context.

    Supported policies:
    - `fp_only`: decode path is forced to full precision.
    - `quantized`: decode path may consume decode-phase quantized configs.

    Why we validate here:
    - This context is shared by all quantized modules during forward.
    - Central validation prevents silent fallback to unintended behavior.
    """

    if policy not in ("fp_only", "quantized"):
        raise ValueError(f"Unsupported decode policy: {policy}")
    _DECODE_POLICY.set(policy)


def get_runtime_decode_policy() -> DecodePolicy:
    """Return decode policy from runtime phase context.

    Defaults to `fp_only` to preserve backward compatibility.
    """

    return _DECODE_POLICY.get()


def infer_runtime_phase_from_hidden_and_cache(
    hidden_states: Any, past_cache: Any
) -> Phase:
    """Infer runtime phase from hidden-state shape and cache state.

    Rules are intentionally unchanged from step-1 attention-local logic:
    1) `past_cache is None` -> `prefill`
    2) `past_cache.get_seq_length() > 0` -> `decode`
    3) Fallback: if current query length is 1 -> `decode`, else `prefill`

    Why this helper exists:
    - The same heuristic must be shared by all Llama decoder layers.
    - We keep it in the shared phase module so hooks and tests stay consistent.
    """

    if past_cache is None:
        return "prefill"

    past_len = 0
    get_seq_length = getattr(past_cache, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            past_len = int(get_seq_length())
        except Exception:  # pragma: no cover - defensive for custom Cache impls
            past_len = 0

    if past_len > 0:
        return "decode"

    q_len = 0
    shape = getattr(hidden_states, "shape", None)
    if shape is not None and len(shape) >= 2:
        q_len = int(shape[-2])
    if q_len == 1:
        return "decode"
    return "prefill"


def extract_decoder_layer_past_cache(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Extract past-cache object from decoder-layer call arguments.

    Compatibility:
    - New HF naming: `past_key_values`
    - Legacy alias: `past_key_value`
    - Positional call fallback: 4th argument for `LlamaDecoderLayer.forward`
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
    return infer_runtime_phase_from_hidden_and_cache(hidden_states, past_cache)
