"""Transformers ABI guard for fused Qwen3-MoE modules."""

from __future__ import annotations

import transformers


SUPPORTED_TRANSFORMERS_VERSION = "5.5.0"


def require_qwen3_moe_fused_abi() -> None:
    """Reject releases whose Qwen3-MoE weight layout is incompatible."""

    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise ImportError(
            "MASE fused Qwen3-MoE quantization requires "
            f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}; found "
            f"{transformers.__version__}. Transformers 4.51 uses a "
            "ModuleList expert ABI and is not compatible."
        )
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeExperts,
            Qwen3MoeTopKRouter,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers==5.5.0 must expose fused Qwen3MoeExperts and "
            "Qwen3MoeTopKRouter"
        ) from exc
    if not all(hasattr(Qwen3MoeExperts, name) for name in ("forward",)):
        raise ImportError("Qwen3MoeExperts does not expose the expected fused ABI")
    if not hasattr(Qwen3MoeTopKRouter, "forward"):
        raise ImportError("Qwen3MoeTopKRouter does not expose the expected router ABI")


require_qwen3_moe_fused_abi()


__all__ = ["SUPPORTED_TRANSFORMERS_VERSION", "require_qwen3_moe_fused_abi"]
