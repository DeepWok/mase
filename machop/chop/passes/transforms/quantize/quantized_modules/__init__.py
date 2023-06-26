# from .add import AddInteger
from .conv1d import (
    Conv1dBlockMinifloat,
    Conv1dInteger,
    Conv1dLog,
    Conv1dMinifloatIEEE,
    Conv1dMinifloatSimple,
    Conv1dBlockFP,
)
from .conv2d import (
    Conv2dBlockMinifloat,
    Conv2dInteger,
    Conv2dLog,
    Conv2dMinifloatIEEE,
    Conv2DMinifloatSimple,
    Conv2dBlockFP,
)
from .linear import (
    LinearBlockMinifloat,
    LinearInteger,
    LinearLog,
    LinearMinifloatIEEE,
    LinearMinifloatSimple,
    LinearBlockFP,
)
from .pool2d import AdaptiveAvgPool2dInteger, AvgPool2dInteger
from .relu import (
    ReLUBlockMinifloat,
    ReLUInteger,
    ReLULog,
    ReLUMinifloatIEEE,
    ReLUMinifloatSimple,
    ReLUBlockFP,
)

quantized_module_map = {
    'conv1d_block_minifloat': Conv1dBlockMinifloat,
    'conv1d_integer': Conv1dInteger,
    'conv1d_log': Conv1dLog,
    'conv1d_minifloat_ieee': Conv1dMinifloatIEEE,
    'conv1d_minifloat_simple': Conv1dMinifloatSimple,
    'conv1d_block_fp': Conv1dBlockFP,

    'conv2d_block_minifloat': Conv2dBlockMinifloat,
    'conv2d_integer': Conv2dInteger,
    'conv2d_log': Conv2dLog,
    'conv2d_minifloat_ieee': Conv2dMinifloatIEEE,
    'conv2d_minifloat_simple': Conv2DMinifloatSimple,
    'conv2d_block_fp': Conv2dBlockFP,

    'linear_block_minifloat': LinearBlockMinifloat,
    'linear_integer': LinearInteger,
    'linear_log': LinearLog,
    'linear_minifloat_ieee': LinearMinifloatIEEE,
    'linear_minifloat_simple': LinearMinifloatSimple,
    'linear_block_fp': LinearBlockFP,

    'adaptive_avg_pool2d_integer': AdaptiveAvgPool2dInteger,
    'avg_pool2d_integer': AvgPool2dInteger,

    'relu_block_minifloat': ReLUBlockMinifloat,
    'relu_integer': ReLUInteger,
    'relu_log': ReLULog,
    'relu_minifloat_ieee': ReLUMinifloatIEEE,
    'relu_minifloat_simple': ReLUMinifloatSimple,
    'relu_block_fp': ReLUBlockFP,
}
