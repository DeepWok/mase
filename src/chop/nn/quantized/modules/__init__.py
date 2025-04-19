from .attention_head import BertSelfAttentionHeadInteger
from .attention import BertSelfAttentionInteger

from .roberta import (
    RobertaSelfAttentionLSQInteger,
    RobertaIntermediateLSQInteger,
    RobertaOutputLSQInteger,
    RobertaClassificationHeadLSQInteger,
    RobertaSelfOutputLSQInteger,
)

from .llama import LlamaAttentionLSQInteger

# from .add import AddInteger
from .conv1d import (
    Conv1dBlockFP,
    Conv1dBlockMinifloat,
    Conv1dInteger,
    Conv1dLog,
    Conv1dBlockLog,
    Conv1dMinifloatDenorm,
    Conv1dMinifloatIEEE,
    Conv1dBinary,
    Conv1dTernary,
)
from .conv2d import (
    Conv2dBinaryResidualSign,
    Conv2dBinaryScaling,
    Conv2dBlockFP,
    Conv2dBlockMinifloat,
    Conv2dInteger,
    Conv2dLog,
    Conv2dBlockLog,
    Conv2dMinifloatDenorm,
    Conv2dMinifloatIEEE,
    Conv2dBinary,
    Conv2dTernary,
    Conv2dLUT,
    Conv2DLogicNets,
)
from .convtranspose2d import (
    ConvTranspose2dBinary,
    ConvTranspose2dInteger,
    ConvTranspose2dLog,
    ConvTranspose2dMinifloatDenorm,
    ConvTranspose2dMinifloatIEEE,
)
from .linear import (
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearInteger,
    LinearLog,
    LinearBlockLog,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearBinary,
    LinearBinaryResidualSign,
    LinearBinaryScaling,
    LinearTernary,
    LinearLUT,
    LinearLogicNets,
    LinearMXIntHardware,
)
from .pool2d import (
    AdaptiveAvgPool2dInteger,
    AvgPool2dInteger,
    AvgPool2dBinary,
    AvgPool2dTernary,
)
from .relu import (
    ReLUBlockFP,
    ReLUBlockMinifloat,
    ReLUInteger,
    ReLULog,
    ReLUBlockLog,
    ReLUMinifloatDenorm,
    ReLUMinifloatIEEE,
    ReLUBinary,
    ReLUTernary,
)
from .batch_norm2d import (
    BatchNorm2dInteger,
    BatchNorm2dBinary,
)
from .layer_norm import (
    LayerNormInteger,
)
from .group_norm import GroupNormInteger
from .instance_norm2d import InstanceNorm2dInteger

from .rms_norm import RMSNormInteger

from .selu import (
    SELUBlockFP,
    SELUBlockMinifloat,
    SELUInteger,
    SELULog,
    SELUBlockLog,
    SELUMinifloatDenorm,
    SELUMinifloatIEEE,
    SELUBinary,
    SELUTernary,
)

from .silu import (
    SiLUBlockFP,
    SiLUBlockMinifloat,
    SiLUInteger,
    SiLULog,
    SiLUBlockLog,
    SiLUMinifloatDenorm,
    SiLUMinifloatIEEE,
    SiLUBinary,
    SiLUTernary,
)

from .tanh import (
    TanhBlockFP,
    TanhBlockMinifloat,
    TanhInteger,
    TanhLog,
    TanhBlockLog,
    TanhMinifloatDenorm,
    TanhMinifloatIEEE,
    TanhBinary,
    TanhTernary,
)

from .gelu import (
    GELUBlockFP,
    GELUBlockMinifloat,
    GELUInteger,
    GELULog,
    GELUBlockLog,
    GELUMinifloatDenorm,
    GELUMinifloatIEEE,
    GELUBinary,
    GELUTernary,
)

from .softsign import (
    SoftsignBlockFP,
    SoftsignBlockMinifloat,
    SoftsignInteger,
    SoftsignLog,
    SoftsignBlockLog,
    SoftsignMinifloatDenorm,
    SoftsignMinifloatIEEE,
    SoftsignBinary,
    SoftsignTernary,
)

from .softplus import (
    SoftplusBlockFP,
    SoftplusBlockMinifloat,
    SoftplusInteger,
    SoftplusLog,
    SoftplusBlockLog,
    SoftplusMinifloatDenorm,
    SoftplusMinifloatIEEE,
    SoftplusBinary,
    SoftplusTernary,
)
from .batch_norm1d import (
    BatchNorm1dInteger,
)
from .gqa import (
    GroupedQueryAttentionInteger,
)

quantized_basic_module_map = {
    "conv1d_block_minifloat": Conv1dBlockMinifloat,
    "conv1d_integer": Conv1dInteger,
    "conv1d_binary": Conv1dBinary,
    "conv1d_ternary": Conv1dTernary,
    "conv1d_log": Conv1dLog,
    "conv1d_block_log": Conv1dBlockLog,
    "conv1d_minifloat_ieee": Conv1dMinifloatIEEE,
    "conv1d_minifloat_denorm": Conv1dMinifloatDenorm,
    "conv1d_block_fp": Conv1dBlockFP,
    "conv2d_block_minifloat": Conv2dBlockMinifloat,
    "conv2d_integer": Conv2dInteger,
    "conv2d_binary_residual": Conv2dBinaryResidualSign,
    "conv2d_binary": Conv2dBinaryScaling,
    "conv2d_ternary": Conv2dTernary,
    "conv2d_log": Conv2dLog,
    "conv2d_block_log": Conv2dBlockLog,
    "conv2d_minifloat_ieee": Conv2dMinifloatIEEE,
    "conv2d_minifloat_denorm": Conv2dMinifloatDenorm,
    "conv2d_block_fp": Conv2dBlockFP,
    "conv2d_lutnet": Conv2dLUT,
    "conv2d_logicnets": Conv2DLogicNets,
    "convtranspose2d_integer": ConvTranspose2dInteger,
    "convtranspose2d_binary": ConvTranspose2dBinary,
    "convtranspose2d_log": ConvTranspose2dLog,
    "convtranspose2d_minifloat_ieee": ConvTranspose2dMinifloatIEEE,
    "convtranspose2d_minifloat_denorm": ConvTranspose2dMinifloatDenorm,
    "linear_block_minifloat": LinearBlockMinifloat,
    "linear_integer": LinearInteger,
    "linear_fixed": LinearInteger,
    "linear_log": LinearLog,
    "linear_mxint_hardware": LinearMXIntHardware,
    "linear_block_log": LinearBlockLog,
    "linear_minifloat_ieee": LinearMinifloatIEEE,
    "linear_minifloat_denorm": LinearMinifloatDenorm,
    "linear_block_fp": LinearBlockFP,
    "linear_binary": LinearBinary,
    "linear_binary_residual": LinearBinaryResidualSign,
    "linear_ternary": LinearTernary,
    "linear_lutnet": LinearLUT,
    "linear_logicnets": LinearLogicNets,
    "adaptive_avg_pool2d_integer": AdaptiveAvgPool2dInteger,
    "avg_pool2d_integer": AvgPool2dInteger,
    "avg_pool2d_binary": AvgPool2dBinary,
    "avg_pool2d_ternary": AvgPool2dTernary,
    "relu_block_minifloat": ReLUBlockMinifloat,
    "relu_integer": ReLUInteger,
    "relu_fixed": ReLUInteger,
    "relu_log": ReLULog,
    "relu_block_log": ReLUBlockLog,
    "relu_minifloat_ieee": ReLUMinifloatIEEE,
    "relu_minifloat_denorm": ReLUMinifloatDenorm,
    "relu_block_fp": ReLUBlockFP,
    "relu_binary": ReLUBinary,
    "relu_ternary": ReLUTernary,
    "batch_norm2d_integer": BatchNorm2dInteger,
    "batch_norm2d_binary": BatchNorm2dBinary,
    "layer_norm_integer": LayerNormInteger,
    "group_norm_integer": GroupNormInteger,
    "instance_norm2d_integer": InstanceNorm2dInteger,
    "rms_norm_integer": RMSNormInteger,
    "selu_block_minifloat": SELUBlockMinifloat,
    "selu_integer": SELUInteger,
    "selu_fixed": SELUInteger,
    "selu_log": SELULog,
    "selu_block_log": SELUBlockLog,
    "selu_minifloat_ieee": SELUMinifloatIEEE,
    "selu_minifloat_denorm": SELUMinifloatDenorm,
    "selu_block_fp": SELUBlockFP,
    "selu_binary": SELUBinary,
    "selu_ternary": SELUTernary,
    "silu_block_minifloat": SiLUBlockMinifloat,
    "silu_integer": SiLUInteger,
    "silu_fixed": SiLUInteger,
    "silu_log": SiLULog,
    "silu_block_log": SiLUBlockLog,
    "silu_minifloat_ieee": SiLUMinifloatIEEE,
    "silu_minifloat_denorm": SiLUMinifloatDenorm,
    "silu_block_fp": SiLUBlockFP,
    "silu_binary": SiLUBinary,
    "silu_ternary": SiLUTernary,
    "tanh_block_minifloat": TanhBlockMinifloat,
    "tanh_integer": TanhInteger,
    "tanh_fixed": TanhInteger,
    "tanh_log": TanhLog,
    "tanh_block_log": TanhBlockLog,
    "tanh_minifloat_ieee": TanhMinifloatIEEE,
    "tanh_minifloat_denorm": TanhMinifloatDenorm,
    "tanh_block_fp": TanhBlockFP,
    "tanh_binary": TanhBinary,
    "tanh_ternary": TanhTernary,
    "gelu_block_minifloat": GELUBlockMinifloat,
    "gelu_integer": GELUInteger,
    "gelu_fixed": GELUInteger,
    "gelu_log": GELULog,
    "gelu_block_log": GELUBlockLog,
    "gelu_minifloat_ieee": GELUMinifloatIEEE,
    "gelu_minifloat_denorm": GELUMinifloatDenorm,
    "gelu_block_fp": GELUBlockFP,
    "gelu_binary": GELUBinary,
    "gelu_ternary": GELUTernary,
    "softsign_block_minifloat": SoftsignBlockMinifloat,
    "softsign_integer": SoftsignInteger,
    "softsign_fixed": SoftsignInteger,
    "softsign_log": SoftsignLog,
    "softsign_block_log": SoftsignBlockLog,
    "softsign_minifloat_ieee": SoftsignMinifloatIEEE,
    "softsign_minifloat_denorm": SoftsignMinifloatDenorm,
    "softsign_block_fp": SoftsignBlockFP,
    "softsign_binary": SoftsignBinary,
    "softsign_ternary": SoftsignTernary,
    "softplus_block_minifloat": SoftplusBlockMinifloat,
    "softplus_integer": SoftplusInteger,
    "softplus_fixed": SoftplusInteger,
    "softplus_log": SoftplusLog,
    "softplus_block_log": SoftplusBlockLog,
    "softplus_minifloat_ieee": SoftplusMinifloatIEEE,
    "softplus_minifloat_denorm": SoftplusMinifloatDenorm,
    "softplus_block_fp": SoftplusBlockFP,
    "softplus_binary": SoftplusBinary,
    "softplus_ternary": SoftplusTernary,
    "batch_norm1d_fixed": BatchNorm1dInteger,
    "batch_norm1d_linear": BatchNorm1dInteger,
}

quantized_bert_module_map = {
    "bert_self_attention_head_integer": BertSelfAttentionHeadInteger,
    "bert_self_attention_integer": BertSelfAttentionInteger,
    "grouped_query_attention_integer": GroupedQueryAttentionInteger,
}

quantized_roberta_module_map = {
    "roberta_self_attention_lsqinteger": RobertaSelfAttentionLSQInteger,
    "roberta_intermediate_lsqinteger": RobertaIntermediateLSQInteger,
    "roberta_output_lsqinteger": RobertaOutputLSQInteger,
    "roberta_classification_head_lsqinteger": RobertaClassificationHeadLSQInteger,
    "roberta_self_output_lsqinteger": RobertaSelfOutputLSQInteger,
}

quantized_llama_module_map = {
    "llama_self_attention_lsqinteger": LlamaAttentionLSQInteger,
}

quantized_module_map = (
    quantized_basic_module_map
    | quantized_bert_module_map
    | quantized_roberta_module_map
    | quantized_llama_module_map
)


from .flexround import LinearFlexRound, Conv2dFlexRound, Conv1dFlexRound

quantized_module_map.update(
    {
        "linear_flexround": LinearFlexRound,
        "conv2d_flexround": Conv2dFlexRound,
        "conv1d_flexround": Conv1dFlexRound,
    }
)
