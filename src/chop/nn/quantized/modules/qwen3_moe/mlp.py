from torch import Tensor

from chop.nn.quantized.functional.silu import silu_minifloat

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP


class Qwen3MoeMLPMXFP(Qwen3MoeMLP):
    """MXFP-quantized Qwen3MoeMLP (individual expert). SiLU uses minifloat quantization."""

    def __init__(
        self, config, intermediate_size=None, layer_idx=None, q_config: dict = None
    ):
        super().__init__(config, intermediate_size)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.bypass = self.q_config.get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        x = silu_minifloat(self.gate_proj(x), self.q_config) * self.up_proj(x)
        return self.down_proj(x)


class Qwen3MoeMLPMXInt(Qwen3MoeMLP):
    """MXInt-quantized Qwen3MoeMLP (individual expert). SiLU uses minifloat quantization."""

    def __init__(
        self, config, intermediate_size=None, layer_idx=None, q_config: dict = None
    ):
        super().__init__(config, intermediate_size)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.bypass = self.q_config.get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        x = silu_minifloat(self.gate_proj(x), self.q_config) * self.up_proj(x)
        return self.down_proj(x)
