# from .add import AddInteger
from .conv1d import (
    Conv1dBlockFP,
    Conv1dBlockMinifloat,
    Conv1dInteger,
    Conv1dLog,
    Conv1dMinifloatDenorm,
    Conv1dMinifloatIEEE,
)
from .conv2d import (
    Conv2dBlockFP,
    Conv2dBlockMinifloat,
    Conv2dInteger,
    Conv2dLog,
    Conv2dMinifloatDenorm,
    Conv2dMinifloatIEEE,
)
from .linear import (
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearInteger,
    LinearLog,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
)
from .pool2d import AdaptiveAvgPool2dInteger, AvgPool2dInteger
from .relu import (
    ReLUBlockFP,
    ReLUBlockMinifloat,
    ReLUInteger,
    ReLULog,
    ReLUMinifloatDenorm,
    ReLUMinifloatIEEE,
)

quantized_module_map = {
    "conv1d_block_minifloat": Conv1dBlockMinifloat,
    "conv1d_integer": Conv1dInteger,
    "conv1d_log": Conv1dLog,
    "conv1d_minifloat_ieee": Conv1dMinifloatIEEE,
    "conv1d_minifloat_denorm": Conv1dMinifloatDenorm,
    "conv1d_block_fp": Conv1dBlockFP,
    "conv2d_block_minifloat": Conv2dBlockMinifloat,
    "conv2d_integer": Conv2dInteger,
    "conv2d_log": Conv2dLog,
    "conv2d_minifloat_ieee": Conv2dMinifloatIEEE,
    "conv2d_minifloat_denorm": Conv2dMinifloatDenorm,
    "conv2d_block_fp": Conv2dBlockFP,
    "linear_block_minifloat": LinearBlockMinifloat,
    "linear_integer": LinearInteger,
    "linear_log": LinearLog,
    "linear_minifloat_ieee": LinearMinifloatIEEE,
    "linear_minifloat_denorm": LinearMinifloatDenorm,
    "linear_block_fp": LinearBlockFP,
    "adaptive_avg_pool2d_integer": AdaptiveAvgPool2dInteger,
    "avg_pool2d_integer": AvgPool2dInteger,
    "relu_block_minifloat": ReLUBlockMinifloat,
    "relu_integer": ReLUInteger,
    "relu_log": ReLULog,
    "relu_minifloat_ieee": ReLUMinifloatIEEE,
    "relu_minifloat_denorm": ReLUMinifloatDenorm,
    "relu_block_fp": ReLUBlockFP,
}
