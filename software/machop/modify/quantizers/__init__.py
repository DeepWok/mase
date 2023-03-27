from typing import Dict, Tuple

from .functions import (
    add_integer,
    add_minifloat_ieee,
    add_minifloat_simple,
    bmm_integer,
    bmm_minifloat_ieee,
    bmm_minifloat_simple,
    matmul_integer,
    matmul_minifloat_ieee,
    matmul_minifloat_simple,
    relu_integer,
    relu_minifloat_ieee,
    relu_minifloat_simple,
)
from .ops import (
    AddInteger,
    Conv1dInteger,
    Conv1dMinifloatIEEE,
    Conv1dMinifloatSimple,
    Conv2dInteger,
    Conv2dMinifloatIEEE,
    Conv2DMinifloatSimple,
    LinearInteger,
    LinearMinifloatIEEE,
    LinearMinifloatSimple,
    LinearMSFP,
    ReLUInteger,
    ReLUMinifloatIEEE,
    ReLUMinifloatSimple,
)

# possible_ops = [LinearInteger, ReLUInteger, Conv2dInteger, AddInteger, LinearMSFP]
# possible_functions = [add_integer]


ops_map = {
    "linear": {
        "integer": LinearInteger,
        "minifloat_simple": LinearMinifloatSimple,
        "minifloat_ieee": LinearMinifloatIEEE,
        "msfp": LinearMSFP,
    },
    "relu": {
        "integer": ReLUInteger,
        "minifloat_simple": ReLUMinifloatSimple,
        "minifloat_ieee": ReLUMinifloatIEEE,
    },
    "conv1d": {
        "integer": Conv1dInteger,
        "minifloat_simple": Conv1dMinifloatSimple,
        "minifloat_ieee": Conv1dMinifloatIEEE,
    },
    "conv2d": {
        "integer": Conv2dInteger,
        "minifloat_simple": Conv2DMinifloatSimple,
        "minifloat_ieee": Conv2dMinifloatIEEE,
    },
    "add": {
        "integer": AddInteger,
    },
}

functions_map = {
    "add": {
        "integer": add_integer,
        "minifloat_simple": add_minifloat_simple,
        "minifloat_ieee": add_minifloat_ieee,
    },
    "relu": {
        "integer": relu_integer,
        "minifloat_simple": relu_minifloat_simple,
        "minifloat_ieee": relu_minifloat_ieee,
    },
    "matmul": {
        "integer": matmul_integer,
        "minifloat_simple": matmul_minifloat_simple,
        "minifloat_ieee": matmul_minifloat_ieee,
    },
    "bmm": {
        "integer": bmm_integer,
        "minifloat_simple": bmm_minifloat_simple,
        "minifloat_ieee": bmm_minifloat_ieee,
    },
}
