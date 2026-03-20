from torch import Tensor

from chop.nn.quantized.functional.silu import silu_minifloat

from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


class Qwen3MLPMXFP(Qwen3MLP):
    """MXFP-quantized Qwen3MLP. SiLU uses minifloat quantization."""

    def __init__(self, config, layer_idx=None, q_config: dict = None):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.bypass = self.q_config.get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        x = silu_minifloat(self.gate_proj(x), self.q_config) * self.up_proj(x)
        return self.down_proj(x)


class Qwen3MLPMXInt(Qwen3MLP):
    """MXInt-quantized Qwen3MLP. SiLU uses minifloat quantization."""

    def __init__(self, config, layer_idx=None, q_config: dict = None):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.bypass = self.q_config.get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        x = silu_minifloat(self.gate_proj(x), self.q_config) * self.up_proj(x)
        return self.down_proj(x)
