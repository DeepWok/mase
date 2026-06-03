"""Shared phase-config normalization helpers for quantized modules.

Step-1 integration contracts:
1. Accept both legacy flat configs and phase-structured configs.
2. Return a stable normalized shape consumed by runtime modules.
3. Do not decide runtime policy here (that remains module-side logic).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_phase_q_config(q_config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize config into `{decode_policy, prefill, decode}` shape.

    Compatibility behavior:
    - Legacy flat config -> copied into both prefill and decode buckets.
    - Explicit phase config -> missing decode bucket falls back to prefill.

    Why we keep this helper minimal:
    - It preserves old config semantics without enforcing model-specific
      policy in the parsing layer.
    """

    cfg = deepcopy(q_config or {})
    decode_policy = cfg.get("decode_policy", "fp_only")

    if "prefill" in cfg or "decode" in cfg:
        prefill = deepcopy(cfg.get("prefill", {}))
        decode = deepcopy(cfg.get("decode", prefill))
    else:
        legacy = deepcopy(cfg)
        legacy.pop("decode_policy", None)
        prefill = legacy
        decode = deepcopy(legacy)

    return {
        "decode_policy": decode_policy,
        "prefill": prefill,
        "decode": decode,
    }


def get_phase_subconfig(
    normalized: dict[str, Any], phase: str
) -> tuple[dict[str, Any], str]:
    """Select phase sub-config from normalized phase config.

    Returns:
        (sub_config, decode_policy)
    """

    if phase == "decode":
        return deepcopy(normalized.get("decode", {})), normalized["decode_policy"]
    return deepcopy(normalized.get("prefill", {})), normalized["decode_policy"]
