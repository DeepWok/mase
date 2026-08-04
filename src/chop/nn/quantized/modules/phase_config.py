"""Phase-config normalization for prefill/decode-split quantisation.

In disaggregated serving (e.g. PLENA) prefill and decode run on separate
chips with independent quantisation profiles: the prefill chip stays
unquantised (weights/activations in the task dtype, bf16/fp16), while the
decode chip runs quantised weights / activations / KV cache (MX formats,
GPTQ, rotation). One emulated model therefore carries a quantisation config
per phase.

Config schema
-------------
A phase-structured q_config looks like::

    {
        "name": "mxint",                    # consumed by the quantize pass
        "kv_cache_handoff": "fp",  # or explicit legacy "decode_format"
        "prefill": { ... flat sub-config ... },
        "decode":  { ... flat sub-config ... },
        "decode_policy": "quantized",       # optional; inferred if omitted
    }

Normalization rules (``normalize_phase_q_config``):

1. Legacy flat config (no ``prefill``/``decode`` keys): copied into BOTH
   buckets and ``decode_policy`` defaults to ``"quantized"`` — a flat config
   keeps quantising both phases identically, with no extra memory.
2. Phase-structured config: a missing ``prefill`` bucket defaults to
   ``{"bypass": True}`` (decode-only shorthand: prefill chip is FP); a
   missing ``decode`` bucket falls back to a copy of ``prefill``.
3. ``decode_policy`` is inferred when omitted: ``"quantized"`` unless the
   decode bucket is empty or a blanket ``{"bypass": True}``. An explicit
   value always wins; ``fp_only`` force-bypasses every decode stage (used
   by prefill-quantised deployments that keep decode at full precision).

KV-cache handoff semantics (``kv_cache_handoff``):

- ``"decode_format"``: KV vectors written during *prefill* are
  quantised with the DECODE bucket's ``kv_cache`` config even when the rest
  of prefill is bypassed.
- ``"fp"`` (default): prefill-produced KV stays in the prefill dtype.
- An explicit ``kv_cache`` entry in the prefill bucket always overrides the
  handoff rule.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Keys that steer the pass / phase machinery rather than any quantizer.
_CONTROL_KEYS = ("name", "decode_policy", "kv_cache_handoff", "prefill", "decode")

# Attribute names of the GPTQ handoff contract. run_gptq writes these on the
# source nn.Linear modules; weight_replacement reads them when building the
# phase-aware replacement. One definition prevents producer/consumer drift.
GPTQ_DECODE_WEIGHT_ATTR = "_mase_gptq_weight_decode"
DECODE_FP_WEIGHT_ATTR = "_mase_decode_weight_fp"
DECODE_FP_BIAS_ATTR = "_mase_decode_bias_fp"

_VALID_DECODE_POLICIES = ("fp_only", "quantized")
_VALID_KV_HANDOFFS = ("decode_format", "fp")

# Attention sub-stages that carry their own nested config dicts.
ATTENTION_STAGES = (
    "qk_norm",
    "qk_matmul",
    "av_matmul",
    "rope",
    "softmax",
    "kv_cache",
)

_BYPASS = {"bypass": True}


def _strip_control_keys(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in _CONTROL_KEYS}


def _is_effectively_bypass(bucket: dict[str, Any]) -> bool:
    """True when a bucket quantises nothing.

    A bucket counts as bypass when it is empty, or sets a blanket
    ``bypass: True`` without re-enabling any nested attention stage.
    """

    if not bucket:
        return True
    if not bucket.get("bypass", False):
        return False
    # Blanket bypass, unless a nested stage explicitly re-enables quantisation
    # (e.g. prefill: {bypass: True, kv_cache: {...}} for KV handoff).
    for stage in ATTENTION_STAGES:
        stage_cfg = bucket.get(stage)
        if isinstance(stage_cfg, dict) and not stage_cfg.get("bypass", False):
            return False
    return True


def normalize_phase_q_config(q_config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a q_config into the stable phase-structured shape.

    Returns ``{decode_policy, kv_cache_handoff, prefill, decode}``. The
    function is idempotent: normalizing an already-normalized config yields
    the same result, so pass-level and module-level normalization compose.
    """

    cfg = deepcopy(q_config or {})

    kv_cache_handoff = cfg.get("kv_cache_handoff", "fp")
    if kv_cache_handoff not in _VALID_KV_HANDOFFS:
        raise ValueError(
            f"Unsupported kv_cache_handoff {kv_cache_handoff!r}; "
            f"expected one of {_VALID_KV_HANDOFFS}."
        )

    if "prefill" in cfg or "decode" in cfg:
        if "prefill" in cfg:
            prefill = deepcopy(cfg["prefill"])
        else:
            # Decode-only shorthand: an explicit decode bucket with no
            # prefill bucket means the prefill chip runs unquantised.
            prefill = dict(_BYPASS)
        decode = deepcopy(cfg.get("decode", prefill))
    else:
        legacy = _strip_control_keys(cfg)
        prefill = legacy
        decode = deepcopy(legacy)

    decode_policy = cfg.get("decode_policy")
    if decode_policy is None:
        decode_policy = "fp_only" if _is_effectively_bypass(decode) else "quantized"
    if decode_policy not in _VALID_DECODE_POLICIES:
        raise ValueError(
            f"Unsupported decode_policy {decode_policy!r}; "
            f"expected one of {_VALID_DECODE_POLICIES}."
        )

    return {
        "decode_policy": decode_policy,
        "kv_cache_handoff": kv_cache_handoff,
        "prefill": prefill,
        "decode": decode,
    }


def get_phase_subconfig(
    normalized: dict[str, Any], phase: str
) -> tuple[dict[str, Any], str]:
    """Return the raw ``(bucket, decode_policy)`` for ``phase``.

    Thin accessor that does NOT apply the ``fp_only`` policy; most callers
    want :func:`resolve_module_phase_config` or :func:`resolve_stage_config`.
    """

    if phase == "decode":
        return deepcopy(normalized.get("decode", {})), normalized["decode_policy"]
    return deepcopy(normalized.get("prefill", {})), normalized["decode_policy"]


def resolve_module_phase_config(
    normalized: dict[str, Any], phase: str
) -> dict[str, Any]:
    """Resolve the effective flat sub-config for a module in ``phase``.

    Applies the ``fp_only`` decode policy (force-bypass) so callers only need
    to consult the returned dict's ``bypass`` flag.
    """

    if phase == "decode" and normalized["decode_policy"] == "fp_only":
        return dict(_BYPASS)
    bucket = normalized["decode"] if phase == "decode" else normalized["prefill"]
    return deepcopy(bucket)


def resolve_stage_config(
    normalized: dict[str, Any], phase: str, stage: str
) -> dict[str, Any]:
    """Resolve one attention sub-stage config for ``phase``.

    Resolution order:
    1. ``fp_only`` decode policy force-bypasses every decode stage.
    2. An explicit stage entry in the phase bucket wins (even under a
       blanket bucket-level ``bypass`` — explicit beats blanket, which is
       how ``prefill: {bypass: True, kv_cache: {...}}`` re-enables the KV
       handoff write).
    3. ``kv_cache`` in prefill with no explicit entry mirrors the DECODE
       bucket's ``kv_cache`` config when ``kv_cache_handoff ==
       "decode_format"`` — prefill KV is quantised into the decode chip's
       storage format at handoff.
    4. A missing stage entry is a bypass (nothing to quantise).
    """

    if stage not in ATTENTION_STAGES:
        raise ValueError(f"Unknown attention stage {stage!r}; valid: {ATTENTION_STAGES}")

    if phase == "decode" and normalized["decode_policy"] == "fp_only":
        return dict(_BYPASS)

    bucket = normalized["decode"] if phase == "decode" else normalized["prefill"]
    stage_cfg = bucket.get(stage)
    if isinstance(stage_cfg, dict):
        return deepcopy(stage_cfg)

    if (
        stage == "kv_cache"
        and phase == "prefill"
        and normalized["kv_cache_handoff"] == "decode_format"
    ):
        # Handoff rule: prefill KV writes land in the decode chip's HBM, so
        # they take the decode-side KV format (including its bypass state).
        return resolve_stage_config(normalized, "decode", "kv_cache")

    return dict(_BYPASS)
