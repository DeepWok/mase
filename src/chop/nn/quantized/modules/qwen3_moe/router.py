"""BF16 router safety island for Qwen3-MoE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

from chop.nn.quantized.functional.vector import VectorRoundingPolicy
from chop.nn.quantized.modules.phase_config import (
    normalize_phase_q_config,
    resolve_module_phase_config,
)
from chop.nn.quantized.modules.phase_context import get_runtime_phase


class Qwen3MoeTopKRouterBF16(Qwen3MoeTopKRouter):
    """Keep router GEMM in BF16 and its probability/top-k path in FP32."""

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(
            hidden_states.to(torch.bfloat16), self.weight.to(torch.bfloat16)
        )
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        router_top_value, router_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )
        return router_probs, router_top_value, router_indices

    @classmethod
    def from_self(cls, router: Qwen3MoeTopKRouter, q_config: dict | None = None):
        del q_config
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.top_k = router.top_k
        new.num_experts = router.num_experts
        new.norm_topk_prob = router.norm_topk_prob
        new.hidden_dim = router.hidden_dim
        # Keep one resident BF16 copy; per-forward weight conversion would be
        # prohibitive in token-by-token decode.
        new.weight = nn.Parameter(
            router.weight.detach().to(torch.bfloat16),
            requires_grad=router.weight.requires_grad,
        )
        return new


class Qwen3MoeSparseMoeBlockBF16Router(Qwen3MoeSparseMoeBlock):
    """Sparse block retaining HF routing semantics with a BF16 router."""

    @classmethod
    def from_self(
        cls, block: Qwen3MoeSparseMoeBlock, q_config: dict | None = None
    ):
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.experts = block.experts
        new.gate = Qwen3MoeTopKRouterBF16.from_self(block.gate, q_config=q_config)
        return new


class Qwen3MoeSparseMoeBlockMinifloat(Qwen3MoeSparseMoeBlockBF16Router):
    """BF16 routing with phase-aware rounding at routed combine."""

    def _init_phase_config(self, q_config: dict | None) -> None:
        self.q_config = q_config or {}
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self._phase_policies = {
            phase: VectorRoundingPolicy.from_config(
                resolve_module_phase_config(self.phase_q_config, phase)
            )
            for phase in ("prefill", "decode")
        }
        self.bypass = not self._phase_policies["decode"].enabled

    def forward(self, hidden_states):
        output = super().forward(hidden_states)
        policy = self._phase_policies[get_runtime_phase()]
        return policy.round(output) if policy.enabled else output

    @classmethod
    def from_self(
        cls, block: Qwen3MoeSparseMoeBlock, q_config: dict | None = None
    ):
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.experts = block.experts
        new.gate = Qwen3MoeTopKRouterBF16.from_self(block.gate, q_config=q_config)
        new._init_phase_config(q_config)
        return new
