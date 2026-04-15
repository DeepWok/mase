from torch import Tensor

from chop.nn.quantized.functional.silu import silu_minifloat

from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMLP


class Glm4MoeMLPMXFP(Glm4MoeMLP):
    """MXFP-quantized Glm4MoeMLP (individual expert). SiLU uses minifloat quantization.

    GLM4 MoE MLP uses separate gate_proj + up_proj (same as Llama).
    """

    def __init__(self, config, intermediate_size=None, layer_idx=None, q_config: dict = None):
        super().__init__(config, intermediate_size=intermediate_size)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.bypass = self.q_config.get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        x = silu_minifloat(self.gate_proj(x), self.q_config) * self.up_proj(x)
        return self.down_proj(x)


class Glm4MoeMLPMXInt(Glm4MoeMLP):
    """MXInt-quantized Glm4MoeMLP (individual expert). SiLU uses minifloat quantization.

    GLM4 MoE MLP uses separate gate_proj + up_proj (same as Llama).
    """

    def __init__(self, config, intermediate_size=None, layer_idx=None, q_config: dict = None):
        super().__init__(config, intermediate_size=intermediate_size)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.bypass = self.q_config.get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        x = silu_minifloat(self.gate_proj(x), self.q_config) * self.up_proj(x)
        return self.down_proj(x)
