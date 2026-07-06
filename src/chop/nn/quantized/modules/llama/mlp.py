import torch
from torch import nn, Tensor

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantized.functional.silu import silu_minifloat
from chop.nn.quantized.modules.phase_context import get_runtime_phase
from chop.nn.quantized.modules.phase_config import (
    normalize_phase_q_config,
    resolve_module_phase_config,
)

from transformers.models.llama.modeling_llama import LlamaMLP, ACT2FN


class LlamaMLPLSQInteger(LlamaMLP):
    def __init__(self, config, layer_idx, q_config: dict = None):
        super().__init__(config)
        self.config = config
        self.q_config = q_config
        self.layer_idx = layer_idx
        # NOTE: The only change from the original RobertaOutput is the quantization of the dense layer
        # Preserving the original layer architecture for state_dict compatibility
        self.gate_dense_quan = LSQInteger(level=q_config["level"], sym=False)
        self.up_dense_quan = LSQInteger(level=q_config["level"], sym=False)
        self.down_dense_quan = LSQInteger(level=q_config["level"], sym=True)

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        up = self.up_proj(x)
        up = self.up_dense_quan(up)
        gate = self.gate_proj(x)
        gate = self.gate_dense_quan(gate)

        down_proj = self.down_proj(self.act_fn(gate) * up)
        down_proj = self.down_dense_quan(down_proj)

        return down_proj


class _PhaseAwareGLUMLPMixin:
    """Phase-aware SiLU quantisation for gated MLPs (Llama / Qwen3).

    The gate/up/down projections are separate ``nn.Linear`` modules replaced
    independently by the quantize pass, so only the SiLU stage lives here.
    A decode-only deployment keeps prefill on the plain HF forward. Combine
    with the HF MLP class of the target architecture.
    """

    def _init_phase_mlp_config(self, layer_idx, q_config: dict) -> None:
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self._phase_cfgs = {
            phase: resolve_module_phase_config(self.phase_q_config, phase)
            for phase in ("prefill", "decode")
        }
        # Legacy attr mirrors the decode side (the quantised chip).
        self.bypass = self._phase_cfgs["decode"].get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        cfg = self._phase_cfgs[get_runtime_phase()]
        if cfg.get("bypass", False):
            return super().forward(x)
        x = silu_minifloat(self.gate_proj(x), cfg) * self.up_proj(x)
        return self.down_proj(x)


class _PhaseAwareLlamaMLP(_PhaseAwareGLUMLPMixin, LlamaMLP):
    def __init__(self, config, layer_idx=None, q_config: dict = None):
        super().__init__(config)
        self._init_phase_mlp_config(layer_idx, q_config)


class LlamaMLPMXFP(_PhaseAwareLlamaMLP):
    """MXFP-quantized LlamaMLP. SiLU uses minifloat quantization."""


class LlamaMLPMXInt(_PhaseAwareLlamaMLP):
    """MXInt-quantized LlamaMLP. SiLU uses minifloat quantization."""
