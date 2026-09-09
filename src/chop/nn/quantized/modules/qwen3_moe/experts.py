"""Phase-aware quantisation for fused Qwen3-MoE expert tensors."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantized.functional.matrix import plena_matrix_product
from chop.nn.quantized.functional.vector import VectorRoundingPolicy
from chop.nn.quantized.modules.phase_config import (
    DECODE_FP_EXPERT_DOWN_ATTR,
    DECODE_FP_EXPERT_GATE_UP_ATTR,
    GPTQ_DECODE_EXPERT_DOWN_ATTR,
    GPTQ_DECODE_EXPERT_GATE_UP_ATTR,
    normalize_phase_q_config,
    resolve_module_phase_config,
)
from chop.nn.quantized.modules.phase_context import get_runtime_phase


class _PhaseAwareQwen3MoeExperts(Qwen3MoeExperts):
    """Qwen3 routed experts with independent prefill and decode weight banks."""

    _WEIGHT_CFG_KEYS: tuple[str, ...] = ()

    def __init__(self, config, q_config: dict | None = None):
        super().__init__(config)
        self._init_phase_state(q_config)
        self._register_phase_banks()

    def _init_phase_state(self, q_config: dict | None) -> None:
        self.q_config = q_config or {}
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self.prefill_config = resolve_module_phase_config(
            self.phase_q_config, "prefill"
        )
        self.decode_config = resolve_module_phase_config(
            self.phase_q_config, "decode"
        )
        self.bypass = self.prefill_config.get("bypass", False)
        self.gptq = self.prefill_config.get("gptq", False)
        self.decode_gptq = self.decode_config.get("gptq", False)
        self.shared_phase_banks = self.prefill_config == self.decode_config
        self._validate_weight_configs()

    def _register_phase_banks(self) -> None:
        self.register_buffer(
            "_decode_gate_up_q", torch.empty(0), persistent=False
        )
        self.register_buffer("_decode_down_q", torch.empty(0), persistent=False)
        self.register_buffer(
            "_decode_gate_up_fp", torch.empty(0), persistent=False
        )
        self.register_buffer("_decode_down_fp", torch.empty(0), persistent=False)
        self._decode_bank_collapsed = False
        self._decode_bank_sealed = False
        self._weight_quantization_events = 0

    def _validate_weight_configs(self) -> None:
        for phase, cfg in (
            ("prefill", self.prefill_config),
            ("decode", self.decode_config),
        ):
            if cfg.get("bypass", False) or cfg.get("gptq", False):
                continue
            present = [key for key in self._WEIGHT_CFG_KEYS if cfg.get(key) is not None]
            if present and len(present) != len(self._WEIGHT_CFG_KEYS):
                missing = [
                    key for key in self._WEIGHT_CFG_KEYS if cfg.get(key) is None
                ]
                raise ValueError(
                    f"{type(self).__name__}: incomplete expert weight config in "
                    f"the {phase} bucket; missing {missing}"
                )

    def _weight_config_present(self, cfg: dict) -> bool:
        return all(cfg.get(key) is not None for key in self._WEIGHT_CFG_KEYS)

    def _quantize_weight(self, weight: Tensor, cfg: dict) -> Tensor:
        raise NotImplementedError

    def _quantize_activation(self, value: Tensor, cfg: dict) -> Tensor:
        block_size = cfg.get("data_in_block_size")
        if block_size is None:
            return value
        if cfg.get("data_in_width") is not None:
            return mxint_quantizer(
                value,
                block_size=block_size,
                element_bits=cfg["data_in_width"],
                block_dim=-1,
            )
        if (
            cfg.get("data_in_exponent_width") is not None
            and cfg.get("data_in_frac_width") is not None
        ):
            return mxfp_quantizer(
                value,
                block_size=block_size,
                element_exp_bits=cfg["data_in_exponent_width"],
                element_frac_bits=cfg["data_in_frac_width"],
                block_dim=-1,
            )
        return value

    @torch.no_grad()
    def _quantize_expert_bank(self, weight: Tensor, cfg: dict) -> Tensor:
        """Quantise one expert at a time to bound peak temporary storage."""

        self._weight_quantization_events += 1
        output = torch.empty_like(weight)
        route = os.environ.get("MASE_PHASE_BANK_DEVICE") or None
        for expert_idx in range(weight.shape[0]):
            source = weight[expert_idx]
            quantized = self._quantize_weight(
                source.to(route) if route and str(source.device) != route else source,
                cfg,
            )
            output[expert_idx].copy_(quantized.to(output.device))
        return output

    @torch.no_grad()
    def _build_phase_weight_banks(self) -> None:
        if self._decode_bank_sealed:
            raise RuntimeError("a sealed decode expert bank cannot be rebuilt")

        device = self.gate_up_proj.device
        self._decode_gate_up_q = torch.empty(0, device=device)
        self._decode_down_q = torch.empty(0, device=device)
        self._decode_gate_up_fp = torch.empty(0, device=device)
        self._decode_down_fp = torch.empty(0, device=device)

        prefill_quantizes = (
            not self.bypass
            and not self.gptq
            and self._weight_config_present(self.prefill_config)
        )
        if not self.shared_phase_banks:
            decode_cfg = self.decode_config
            if (
                not decode_cfg.get("bypass", False)
                and not self.decode_gptq
                and self._weight_config_present(decode_cfg)
            ):
                self._decode_gate_up_q = self._quantize_expert_bank(
                    self.gate_up_proj.data, decode_cfg
                )
                self._decode_down_q = self._quantize_expert_bank(
                    self.down_proj.data, decode_cfg
                )
            elif decode_cfg.get("bypass", False) and prefill_quantizes:
                self._decode_gate_up_fp = self.gate_up_proj.detach().clone()
                self._decode_down_fp = self.down_proj.detach().clone()

        if prefill_quantizes:
            self.gate_up_proj.data.copy_(
                self._quantize_expert_bank(
                    self.gate_up_proj.data, self.prefill_config
                )
            )
            self.down_proj.data.copy_(
                self._quantize_expert_bank(self.down_proj.data, self.prefill_config)
            )

    @torch.no_grad()
    def adopt_decode_gptq_expert_weights(
        self, gate_up_proj: Tensor, down_proj: Tensor
    ) -> None:
        self._decode_gate_up_q = gate_up_proj.detach().to(
            device=self.gate_up_proj.device,
            dtype=self.gate_up_proj.dtype,
            copy=True,
        )
        self._decode_down_q = down_proj.detach().to(
            device=self.down_proj.device,
            dtype=self.down_proj.dtype,
            copy=True,
        )

    @torch.no_grad()
    def adopt_decode_fp_expert_weights(
        self, gate_up_proj: Tensor, down_proj: Tensor
    ) -> None:
        self._decode_gate_up_fp = gate_up_proj.detach().to(
            device=self.gate_up_proj.device,
            dtype=self.gate_up_proj.dtype,
            copy=True,
        )
        self._decode_down_fp = down_proj.detach().to(
            device=self.down_proj.device,
            dtype=self.down_proj.dtype,
            copy=True,
        )

    @classmethod
    def from_self(cls, experts: Qwen3MoeExperts, q_config: dict | None = None):
        """Create a wrapper without allocating another pair of expert Parameters."""

        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.num_experts = experts.num_experts
        new.hidden_dim = experts.hidden_dim
        new.intermediate_dim = experts.intermediate_dim
        new.gate_up_proj = experts.gate_up_proj
        new.down_proj = experts.down_proj
        new.act_fn = experts.act_fn
        new._init_phase_state(q_config)
        new._register_phase_banks()
        new._build_phase_weight_banks()

        gate_up_gptq = getattr(experts, GPTQ_DECODE_EXPERT_GATE_UP_ATTR, None)
        down_gptq = getattr(experts, GPTQ_DECODE_EXPERT_DOWN_ATTR, None)
        if gate_up_gptq is not None or down_gptq is not None:
            if gate_up_gptq is None or down_gptq is None:
                raise RuntimeError("incomplete Qwen3-MoE GPTQ decode bank")
            new.adopt_decode_gptq_expert_weights(gate_up_gptq, down_gptq)

        gate_up_fp = getattr(experts, DECODE_FP_EXPERT_GATE_UP_ATTR, None)
        down_fp = getattr(experts, DECODE_FP_EXPERT_DOWN_ATTR, None)
        if gate_up_fp is not None or down_fp is not None:
            if gate_up_fp is None or down_fp is None:
                raise RuntimeError("incomplete Qwen3-MoE FP decode snapshot")
            new.adopt_decode_fp_expert_weights(gate_up_fp, down_fp)
        return new

    @torch.no_grad()
    def collapse_to_decode_bank(self) -> bool:
        if self._decode_gate_up_q.numel() == 0:
            return False
        if self._decode_down_q.numel() == 0:
            raise RuntimeError("incomplete Qwen3-MoE decode bank")
        self.gate_up_proj.data.copy_(self._decode_gate_up_q)
        self.down_proj.data.copy_(self._decode_down_q)
        device = self.gate_up_proj.device
        self._decode_gate_up_q = torch.empty(0, device=device)
        self._decode_down_q = torch.empty(0, device=device)
        self._decode_bank_collapsed = True
        return True

    def seal_decode_weight_bank(self) -> int:
        if not self._decode_bank_collapsed:
            raise RuntimeError("decode expert bank was not collapsed")
        if self._decode_gate_up_q.numel() or self._decode_down_q.numel():
            raise RuntimeError("decode expert bank still has an auxiliary copy")
        self._decode_bank_sealed = True
        return self._weight_quantization_events

    def _active_weights(self) -> tuple[Tensor, Tensor, dict]:
        if get_runtime_phase() != "decode" or self.shared_phase_banks:
            return self.gate_up_proj, self.down_proj, self.prefill_config
        if self._decode_gate_up_q.numel():
            return self._decode_gate_up_q, self._decode_down_q, self.decode_config
        if self._decode_gate_up_fp.numel():
            return self._decode_gate_up_fp, self._decode_down_fp, self.decode_config
        return self.gate_up_proj, self.down_proj, self.decode_config

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        gate_up_bank, down_bank, cfg = self._active_weights()
        if cfg.get("bypass", False):
            # Calling the HF implementation would read self Parameters instead
            # of an FP decode snapshot, so retain the shared routed loop here.
            quantize = False
        else:
            quantize = True

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        vector_token = cfg.get("output_format") or cfg.get("vector_format")
        vector_policy = (
            VectorRoundingPolicy.from_token(vector_token)
            if isinstance(vector_token, str)
            else VectorRoundingPolicy.disabled()
        )
        if self._use_gathered_dispatch(top_k_index):
            return self._forward_gathered(
                hidden_states,
                top_k_index,
                top_k_weights,
                gate_up_bank,
                down_bank,
                cfg,
                quantize,
                vector_policy,
            )
        for expert_hit_idx in expert_hit:
            expert_idx = int(expert_hit_idx[0].item())
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            if quantize:
                current_state = self._quantize_activation(current_state, cfg)
            gate_up = plena_matrix_product(
                current_state,
                gate_up_bank[expert_idx].transpose(-1, -2),
                cfg,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden = (
                vector_policy.silu_gate(gate, up)
                if vector_policy.enabled
                else self.act_fn(gate) * up
            )
            if quantize:
                current_hidden = self._quantize_activation(current_hidden, cfg)
            current_hidden = plena_matrix_product(
                current_hidden,
                down_bank[expert_idx].transpose(-1, -2),
                cfg,
            )
            current_hidden = current_hidden * top_k_weights[
                token_idx, top_k_pos, None
            ]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    # ------------------------------------------------------------------
    # Gathered dispatch for decode-sized token counts.
    #
    # The per-expert loop above issues ~70 small kernels per hit expert, and a
    # q_len=1 microbatch of 16 hits ~100 of the 128 experts in every layer, so
    # one decode step costs hundreds of thousands of launches. The gathered
    # path evaluates every (token, expert) assignment at once: the activation
    # quantiser works per row along ``block_dim=-1`` and ``plena_matrix_product``
    # partitions the reduction dimension per row, so both are numerically the
    # same operations applied to stacked rows. The only semantic difference is
    # the combine: the loop accumulates expert partials into the BF16 output in
    # expert-index order, the gathered path sums the weighted partials of each
    # token in FP32 and rounds once. ``MASE_MOE_EXPERT_DISPATCH=loop`` restores
    # the original loop; ``gather`` forces the gathered path; the default uses
    # the gathered path below ``MASE_MOE_GATHER_MAX_ASSIGNMENTS`` assignments
    # (prefill-sized token counts keep the loop, whose per-expert matmuls are
    # already large).
    # ------------------------------------------------------------------
    _GATHER_CHUNK = 512

    def _use_gathered_dispatch(self, top_k_index: Tensor) -> bool:
        mode = os.environ.get("MASE_MOE_EXPERT_DISPATCH", "auto")
        if mode == "loop":
            return False
        if mode == "gather":
            return True
        limit = int(os.environ.get("MASE_MOE_GATHER_MAX_ASSIGNMENTS", "4096"))
        return top_k_index.numel() <= limit

    def _forward_gathered(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
        gate_up_bank: Tensor,
        down_bank: Tensor,
        cfg: dict,
        quantize: bool,
        vector_policy: VectorRoundingPolicy,
    ) -> Tensor:
        tokens, top_k = top_k_index.shape
        assignment_expert = top_k_index.reshape(-1)
        assignment_token = torch.arange(
            tokens, device=hidden_states.device
        ).repeat_interleave(top_k)
        assignment_weight = top_k_weights.reshape(-1)
        accumulator = torch.zeros(
            hidden_states.shape, dtype=torch.float32, device=hidden_states.device
        )
        total = assignment_expert.numel()
        for start in range(0, total, self._GATHER_CHUNK):
            stop = min(total, start + self._GATHER_CHUNK)
            experts = assignment_expert[start:stop]
            token_idx = assignment_token[start:stop]
            current_state = hidden_states[token_idx]
            if quantize:
                current_state = self._quantize_activation(current_state, cfg)
            gate_up = plena_matrix_product(
                current_state.unsqueeze(1),
                gate_up_bank[experts].transpose(-1, -2),
                cfg,
            ).squeeze(1)
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden = (
                vector_policy.silu_gate(gate, up)
                if vector_policy.enabled
                else self.act_fn(gate) * up
            )
            if quantize:
                current_hidden = self._quantize_activation(current_hidden, cfg)
            current_hidden = plena_matrix_product(
                current_hidden.unsqueeze(1),
                down_bank[experts].transpose(-1, -2),
                cfg,
            ).squeeze(1)
            current_hidden = current_hidden.to(torch.float32) * assignment_weight[
                start:stop, None
            ].to(torch.float32)
            accumulator.index_add_(0, token_idx, current_hidden)
        return accumulator.to(hidden_states.dtype)


class Qwen3MoeExpertsMXFP(_PhaseAwareQwen3MoeExperts):
    _WEIGHT_CFG_KEYS = (
        "weight_block_size",
        "weight_exponent_width",
        "weight_frac_width",
    )

    def _quantize_weight(self, weight: Tensor, cfg: dict) -> Tensor:
        return mxfp_quantizer(
            weight,
            block_size=cfg["weight_block_size"],
            element_exp_bits=cfg["weight_exponent_width"],
            element_frac_bits=cfg["weight_frac_width"],
            block_dim=1,
        )


class Qwen3MoeExpertsMXInt(_PhaseAwareQwen3MoeExperts):
    _WEIGHT_CFG_KEYS = ("weight_block_size", "weight_width")

    def _quantize_weight(self, weight: Tensor, cfg: dict) -> Tensor:
        return mxint_quantizer(
            weight,
            block_size=cfg["weight_block_size"],
            element_bits=cfg["weight_width"],
            block_dim=1,
            quantile_search=cfg.get("clip_search", False),
        )
