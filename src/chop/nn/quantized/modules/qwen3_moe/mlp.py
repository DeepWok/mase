from chop.nn.quantized.modules.llama.mlp import _PhaseAwareGLUMLPMixin

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP


class _PhaseAwareQwen3MoeMLP(_PhaseAwareGLUMLPMixin, Qwen3MoeMLP):
    def __init__(
        self, config, intermediate_size=None, layer_idx=None, q_config: dict = None
    ):
        super().__init__(config, intermediate_size)
        self._init_phase_mlp_config(layer_idx, q_config)


class Qwen3MoeMLPMXFP(_PhaseAwareQwen3MoeMLP):
    """Phase-aware vector rounding for a dense Qwen3-MoE MLP layer."""


class Qwen3MoeMLPMXInt(_PhaseAwareQwen3MoeMLP):
    """Phase-aware vector rounding for a dense Qwen3-MoE MLP layer."""
