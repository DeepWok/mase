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
    BatchNorm2dInteger
)
from .layer_norm import (
    LayerNormInteger,
)
from .group_norm import (
    GroupNormInteger
)
from .instance_norm2d import (
    InstanceNorm2dInteger
)
# from .rms_norm import (
#     RMSNormInteger
# )

quantized_module_map = {
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
    "linear_block_minifloat": LinearBlockMinifloat,
    "linear_integer": LinearInteger,
    "linear_fixed": LinearInteger,
    "linear_log": LinearLog,
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
    "layer_norm_integer": LayerNormInteger,
    "group_norm_integer": GroupNormInteger,
    "instance_norm2d_integer": InstanceNorm2dInteger,
    # "rms_norm_integer": RMSNormInteger,
}
