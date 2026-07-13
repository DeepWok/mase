import torch
from torch import Tensor, nn

from chop.nn.quantized.modules.phase_context import get_runtime_phase
from chop.nn.quantized.modules.phase_config import (
    normalize_phase_q_config,
    resolve_module_phase_config,
)
from chop.nn.quantized.modules.llama.rms_norm import build_minifloat_phase_quantizers

from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class Qwen3RMSNormMinifloat(Qwen3RMSNorm):
    """Minifloat-quantized Qwen3RMSNorm with per-phase quantizers.

    Quantizers are built once per phase at construction (not per forward
    call, which matters during token-by-token decoding).
    """

    def __init__(self, config=None, layer_idx=None, q_config: dict = None):
        super().__init__(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.variance_epsilon = config.rms_norm_eps
        self.phase_q_config = normalize_phase_q_config(q_config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self._phase_quantizers = {
            phase: build_minifloat_phase_quantizers(
                resolve_module_phase_config(self.phase_q_config, phase)
            )
            for phase in ("prefill", "decode")
        }
        # Legacy attr mirrors the decode side (the quantised chip).
        self.bypass = resolve_module_phase_config(self.phase_q_config, "decode").get(
            "bypass", False
        )

    def forward(self, hidden_states):
        w_quantizer, x_quantizer = self._phase_quantizers[get_runtime_phase()]
        input_dtype = hidden_states.dtype
        if x_quantizer is not None:
            hidden_states = x_quantizer(hidden_states)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = w_quantizer(self.weight) if w_quantizer is not None else self.weight
        # Cast the whole product: module replacement can leave ``self.weight`` in
        # float32 (constructed before the bf16 model context), and multiplying a
        # float32 weight by a bf16 tensor promotes the result back to float32 —
        # which then breaks the next linear (float32 act vs bf16 weight).
        return (weight * hidden_states.to(input_dtype)).to(input_dtype)
