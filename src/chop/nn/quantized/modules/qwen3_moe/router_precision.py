"""Phase-aware MX router ablations for fused Qwen3-MoE blocks.

The canonical Qwen3-MoE path keeps routing in BF16.  These modules form a
separate numerical-ablation lane: only the router matrix operands are
quantized during decode, while softmax, top-k selection, and probability
renormalization retain the Hugging Face FP32 semantics.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from chop.nn.quantized.functional.matrix import plena_matrix_product
from chop.nn.quantized.modules.phase_config import (
    normalize_phase_q_config,
    resolve_module_phase_config,
)
from chop.nn.quantized.modules.phase_context import get_runtime_phase
from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer


ROUTER_PRECISION_SCHEMA = "mase-qwen3-moe-router-mx/v1"
ROUTER_MX_FORMATS = ("MXINT8", "E4M3", "E5M2")
ROUTER_BLOCK_SIZE = 8
ROUTER_OUTPUT_FORMAT = "BF16"


def _canonical_format(value: Any, *, label: str) -> str:
    token = str(value).upper()
    if token not in ROUTER_MX_FORMATS:
        raise ValueError(
            f"{label} must be one of {ROUTER_MX_FORMATS}; observed {value!r}"
        )
    return token


def _format_bits(token: str) -> tuple[str, int, int | None]:
    if token == "MXINT8":
        return "mxint", 8, None
    exponent, fraction = token.removeprefix("E").split("M", maxsplit=1)
    return "mxfp", int(exponent), int(fraction)


def _quantize(value: Tensor, token: str, *, block_dim: int) -> Tensor:
    family, first, second = _format_bits(token)
    if family == "mxint":
        return mxint_quantizer(
            value,
            block_size=ROUTER_BLOCK_SIZE,
            element_bits=first,
            block_dim=block_dim,
        )
    assert second is not None
    return mxfp_quantizer(
        value,
        block_size=ROUTER_BLOCK_SIZE,
        element_exp_bits=first,
        element_frac_bits=second,
        block_dim=block_dim,
    )


def router_decode_config(
    weight_format: str,
    activation_format: str,
    *,
    matrix_mlen: int,
) -> dict[str, Any]:
    """Return the sealed decode bucket for one router-only ablation."""

    if (
        isinstance(matrix_mlen, bool)
        or not isinstance(matrix_mlen, int)
        or matrix_mlen <= 0
        or matrix_mlen % ROUTER_BLOCK_SIZE
    ):
        raise ValueError("matrix_mlen must be a positive multiple of 8")
    return {
        "router_weight_format": _canonical_format(
            weight_format, label="router_weight_format"
        ),
        "router_activation_format": _canonical_format(
            activation_format, label="router_activation_format"
        ),
        "router_block_size": ROUTER_BLOCK_SIZE,
        "matrix_mlen": matrix_mlen,
        "output_format": ROUTER_OUTPUT_FORMAT,
        "matrix_partial_format": ROUTER_OUTPUT_FORMAT,
        "matrix_partial_to_accumulator": "truncate_to_signed_fixed16_16",
        "matrix_cross_partition_accumulation": "signed_fixed16_16_wrap",
        "matrix_final_writeout": ROUTER_OUTPUT_FORMAT,
        "router_logits_container": "BF16",
        "router_probability_dtype": "FP32",
        "router_selection": "exact_torch_topk_sorted",
        "router_topk_renormalization_dtype": "FP32",
    }


def router_phase_config(
    weight_format: str,
    activation_format: str,
    *,
    matrix_mlen: int,
) -> dict[str, Any]:
    """Build the disaggregated prefill-BF16/decode-MX router config."""

    return {
        "decode_policy": "quantized",
        "prefill": {"bypass": True, "router_dtype": "BF16"},
        "decode": router_decode_config(
            weight_format,
            activation_format,
            matrix_mlen=matrix_mlen,
        ),
    }


class Qwen3MoeTopKRouterMX(Qwen3MoeTopKRouter):
    """Decode-only MX router matrix followed by the exact FP32 route path."""

    fixed_weight_format: str | None = None
    fixed_activation_format: str | None = None

    def _init_phase_state(self, q_config: dict[str, Any] | None) -> None:
        self.q_config = deepcopy(q_config or {})
        self.phase_q_config = normalize_phase_q_config(self.q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self.prefill_config = resolve_module_phase_config(
            self.phase_q_config, "prefill"
        )
        self.decode_config = resolve_module_phase_config(
            self.phase_q_config, "decode"
        )
        if not self.prefill_config.get("bypass", False):
            raise ValueError("router MX ablation requires a BF16-bypassed prefill")
        if self.decode_policy != "quantized" or self.decode_config.get(
            "bypass", False
        ):
            raise ValueError("router MX ablation requires quantized decode")

        weight_token = self.fixed_weight_format or self.decode_config.get(
            "router_weight_format"
        )
        activation_token = self.fixed_activation_format or self.decode_config.get(
            "router_activation_format"
        )
        self.router_weight_format = _canonical_format(
            weight_token, label="router_weight_format"
        )
        self.router_activation_format = _canonical_format(
            activation_token, label="router_activation_format"
        )
        if self.fixed_weight_format is not None and self.decode_config.get(
            "router_weight_format", self.fixed_weight_format
        ) != self.fixed_weight_format:
            raise ValueError("router variant weight format conflicts with its config")
        if self.fixed_activation_format is not None and self.decode_config.get(
            "router_activation_format", self.fixed_activation_format
        ) != self.fixed_activation_format:
            raise ValueError("router variant activation format conflicts with its config")

        if self.decode_config.get("router_block_size") != ROUTER_BLOCK_SIZE:
            raise ValueError("router_block_size must equal the PLENA block size 8")
        if self.decode_config.get("output_format") != ROUTER_OUTPUT_FORMAT:
            raise ValueError("router matrix output format must be BF16")
        matrix_mlen = self.decode_config.get("matrix_mlen")
        if (
            isinstance(matrix_mlen, bool)
            or not isinstance(matrix_mlen, int)
            or matrix_mlen <= 0
            or matrix_mlen % ROUTER_BLOCK_SIZE
        ):
            raise ValueError("router matrix_mlen must be a positive multiple of 8")
        for key, required in (
            ("matrix_partial_format", "BF16"),
            (
                "matrix_partial_to_accumulator",
                "truncate_to_signed_fixed16_16",
            ),
            (
                "matrix_cross_partition_accumulation",
                "signed_fixed16_16_wrap",
            ),
            ("matrix_final_writeout", "BF16"),
            ("router_logits_container", "BF16"),
            ("router_probability_dtype", "FP32"),
            ("router_selection", "exact_torch_topk_sorted"),
            ("router_topk_renormalization_dtype", "FP32"),
        ):
            if self.decode_config.get(key) != required:
                raise ValueError(f"router decode config requires {key}={required}")
        self.matrix_mlen = matrix_mlen
        self.bypass = False

    @torch.no_grad()
    def _build_decode_weight_bank(self) -> None:
        self._decode_weight_q = _quantize(
            self.weight.detach(), self.router_weight_format, block_dim=1
        )

    @property
    def router_precision_contract(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTER_PRECISION_SCHEMA,
            "scope": "router_only_decode_ablation",
            "prefill_router": "BF16",
            "decode_router_weight_format": self.router_weight_format,
            "decode_router_activation_format": self.router_activation_format,
            "block_size": ROUTER_BLOCK_SIZE,
            "matrix_mlen": self.matrix_mlen,
            "matrix_arithmetic_chain": [
                "per_mlen_fp32_matmul",
                "bf16_partial_rounding",
                "truncate_partial_to_signed_fixed16_16",
                "signed_fixed16_16_wrap_across_partitions",
                "final_bf16_writeout",
            ],
            "router_logits_container": "BF16",
            "softmax_dtype": "FP32",
            "topk": "torch.topk_sorted_exact",
            "renormalization_dtype": "FP32",
            "publication_rankable": False,
            "selection_eligible": False,
            "hardware_evidence_present": False,
            "compiler_evidence_present": False,
        }

    def _route(self, logits_bf16: Tensor):
        router_probs = F.softmax(logits_bf16, dtype=torch.float32, dim=-1)
        router_top_value, router_indices = torch.topk(
            router_probs, self.top_k, dim=-1, sorted=True
        )
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )
        return router_probs, router_top_value, router_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        if get_runtime_phase() != "decode":
            logits_bf16 = F.linear(
                hidden_states.to(torch.bfloat16), self.weight
            ).to(torch.bfloat16)
            return self._route(logits_bf16)

        activation_q = _quantize(
            hidden_states.to(torch.bfloat16),
            self.router_activation_format,
            block_dim=-1,
        )
        # The matrix oracle rounds every MLEN partial to BF16 before the
        # fixed16.16 bank, wraps cross-partition accumulation, and performs a
        # final BF16 writeout. Probability and selection then execute in FP32.
        logits_bf16 = plena_matrix_product(
            activation_q,
            self._decode_weight_q.transpose(-1, -2),
            {
                "matrix_mlen": self.matrix_mlen,
                "output_format": ROUTER_OUTPUT_FORMAT,
            },
        ).to(torch.bfloat16)
        return self._route(logits_bf16)

    @classmethod
    def from_self(
        cls,
        router: Qwen3MoeTopKRouter,
        q_config: dict[str, Any] | None = None,
    ):
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.top_k = router.top_k
        new.num_experts = router.num_experts
        new.norm_topk_prob = router.norm_topk_prob
        new.hidden_dim = router.hidden_dim
        new.weight = nn.Parameter(
            router.weight.detach().to(torch.bfloat16),
            requires_grad=router.weight.requires_grad,
        )
        new._init_phase_state(q_config)
        new.register_buffer("_decode_weight_q", torch.empty(0), persistent=False)
        new._build_decode_weight_bank()
        return new


class Qwen3MoeTopKRouterMXInt8(Qwen3MoeTopKRouterMX):
    """Uniform MXINT8 router operand ablation."""

    fixed_weight_format = "MXINT8"
    fixed_activation_format = "MXINT8"


class Qwen3MoeTopKRouterE4M3(Qwen3MoeTopKRouterMX):
    """Uniform MXFP E4M3 router operand ablation."""

    fixed_weight_format = "E4M3"
    fixed_activation_format = "E4M3"


class Qwen3MoeTopKRouterE5M2(Qwen3MoeTopKRouterMX):
    """Uniform MXFP E5M2 router operand ablation."""

    fixed_weight_format = "E5M2"
    fixed_activation_format = "E5M2"


__all__ = [
    "Qwen3MoeTopKRouterE4M3",
    "Qwen3MoeTopKRouterE5M2",
    "Qwen3MoeTopKRouterMX",
    "Qwen3MoeTopKRouterMXInt8",
    "ROUTER_BLOCK_SIZE",
    "ROUTER_MX_FORMATS",
    "ROUTER_OUTPUT_FORMAT",
    "ROUTER_PRECISION_SCHEMA",
    "router_decode_config",
    "router_phase_config",
]
