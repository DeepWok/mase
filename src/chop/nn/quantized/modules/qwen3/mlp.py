from torch import Tensor

from chop.nn.quantized.modules.llama.mlp import _PhaseAwareGLUMLPMixin

from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


class _PhaseAwareQwen3MLP(_PhaseAwareGLUMLPMixin, Qwen3MLP):
    def __init__(self, config, layer_idx=None, q_config: dict = None):
        super().__init__(config)
        self._init_phase_mlp_config(layer_idx, q_config)


class Qwen3MLPMXFP(_PhaseAwareQwen3MLP):
    """MXFP-quantized Qwen3MLP. SiLU uses minifloat quantization."""


class Qwen3MLPMXInt(_PhaseAwareQwen3MLP):
    """MXInt-quantized Qwen3MLP. SiLU uses minifloat quantization."""
