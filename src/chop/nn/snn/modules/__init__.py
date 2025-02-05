from torch import nn

from .modules import VoltageHook, VoltageScaler, SeqToANNContainer, MultiStepContainer

# from .converter import Converter

from .surrogate import Sigmoid, ATan

from .conv1d import Conv1d

from .conv2d import Conv2d

from .conv3d import Conv3d

from .linear import Linear, LinearUnfoldBias

from .pool1d import MaxPool1d, AvgPool1d, AdaptiveAvgPool1d

from .pool2d import MaxPool2d, AvgPool2d, AdaptiveAvgPool2d

from .pool3d import MaxPool3d, AvgPool3d, AdaptiveAvgPool3d

from .batch_norm1d import BatchNorm1d

from .batch_norm2d import BatchNorm2d

from .batch_norm3d import BatchNorm3d

from .flatten import Flatten

from .group_norm import GroupNorm

from .upsample import Upsample

from .neuron import (
    IFNode,
    LIFNode,
    ParametricLIFNode,
    ST_BIFNode,
)

from .layernorm import LayerNormZIPTF

from .softmax import SoftmaxZIPTF

from .gelu import GELUZIPTF

from .silu import SiLUZIPTF

from .spiking_self_attention import (
    DSSA,
    GWFFN,
    BN,
    DownsampleLayer,
    Conv1x1,
    LIF,
    PLIF,
    Conv3x3,
    SpikingMatmul,
)

from .embedding import EmbeddingZIPTF
from .roberta import (
    RobertaSelfAttentionZIPTF,
)

spiking_basic_module_map = {
    "conv1d": Conv1d,
    "conv2d": Conv2d,
    "conv3d": Conv3d,
    "linear": Linear,
    "linear_unfold_bias": LinearUnfoldBias,
    "max_pool1d": MaxPool1d,
    "avg_pool1d": AvgPool1d,
    "adaptive_avg_pool1d": AdaptiveAvgPool1d,
    "max_pool2d": MaxPool2d,
    "avg_pool2d": AvgPool2d,
    "adaptive_avg_pool2d": AdaptiveAvgPool2d,
    "max_pool3d": MaxPool3d,
    "avg_pool3d": AvgPool3d,
    "adaptive_avg_pool3d": AdaptiveAvgPool3d,
    "batch_norm1d": BatchNorm1d,
    "batch_norm2d": BatchNorm2d,
    "batch_norm3d": BatchNorm3d,
    "flatten": Flatten,
    "group_norm": GroupNorm,
    "upsample": Upsample,
    "identity": nn.Identity,
}

spiking_varied_module_map = {
    "softmax_zip_tf": SoftmaxZIPTF,
    "layernorm_zip_tf": LayerNormZIPTF,
    "embedding_zip_tf": EmbeddingZIPTF,
    "gelu_zip_tf": GELUZIPTF,
    "silu_zip_tf": SiLUZIPTF,
}

spiking_neuron_module_map = {
    "if": IFNode,
    "lif": LIFNode,
    "plif": ParametricLIFNode,
    "st_bif": ST_BIFNode,
}

spiking_roberta_module_map = {
    "roberta_self_attention_zip_tf": RobertaSelfAttentionZIPTF,
}

spiking_module_map = {
    **spiking_basic_module_map,
    **spiking_neuron_module_map,
    **spiking_varied_module_map,
    **spiking_roberta_module_map,
}
