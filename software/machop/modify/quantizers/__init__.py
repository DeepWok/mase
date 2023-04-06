import operator
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import (
    add_integer,
    add_minifloat_ieee,
    add_minifloat_simple,
    add_msfp,
    bmm_integer,
    bmm_minifloat_ieee,
    bmm_minifloat_simple,
    bmm_msfp,
    matmul_integer,
    matmul_minifloat_ieee,
    matmul_minifloat_simple,
    matmul_msfp,
    relu_integer,
    relu_minifloat_ieee,
    relu_minifloat_simple,
    relu_msfp,
)
from .layers import (
    AddInteger,
    Conv1dInteger,
    Conv1dMinifloatIEEE,
    Conv1dMinifloatSimple,
    Conv1dMSFP,
    Conv2dInteger,
    Conv2dMinifloatIEEE,
    Conv2DMinifloatSimple,
    Conv2dMSFP,
    LinearInteger,
    LinearMinifloatIEEE,
    LinearMinifloatSimple,
    LinearMSFP,
    ReLUInteger,
    ReLUMinifloatIEEE,
    ReLUMinifloatSimple,
    ReLUMSFP,
)

MODULE_CLS_MAP_NEO = {
    nn.Linear: {
        "integer": LinearInteger,
        "minifloat_simple": LinearMinifloatSimple,
        "minifloat_ieee": LinearMinifloatIEEE,
        "msfp": LinearMSFP,
    },
    nn.ReLU: {
        "integer": ReLUInteger,
        "minifloat_simple": ReLUMinifloatSimple,
        "minifloat_ieee": ReLUMinifloatIEEE,
        "msfp": ReLUMSFP,
    },
    nn.Conv1d: {
        "integer": Conv1dInteger,
        "minifloat_simple": Conv1dMinifloatSimple,
        "minifloat_ieee": Conv1dMinifloatIEEE,
        "msfp": Conv1dMSFP,
    },
    nn.Conv2d: {
        "integer": Conv2dInteger,
        "minifloat_simple": Conv2DMinifloatSimple,
        "minifloat_ieee": Conv2dMinifloatIEEE,
        "msfp": Conv2dMSFP,
    },
}

QUANTIZED_MODULE_CLASSES = []
for k, v in MODULE_CLS_MAP_NEO.items():
    for kk, vv in v.items():
        QUANTIZED_MODULE_CLASSES.append(vv)

FUNC_MAP_NEO = {
    operator.add: {
        "integer": add_integer,
        "minifloat_simple": add_minifloat_simple,
        "minifloat_ieee": add_minifloat_ieee,
        "msfp": add_msfp,
    },
    torch.add: {
        "integer": add_integer,
        "minifloat_simple": add_minifloat_simple,
        "minifloat_ieee": add_minifloat_ieee,
        "msfp": add_msfp,
    },
    F.relu: {
        "integer": relu_integer,
        "minifloat_simple": relu_minifloat_simple,
        "minifloat_ieee": relu_minifloat_ieee,
        "msfp": relu_msfp,
    },
    operator.matmul: {
        "integer": matmul_integer,
        "minifloat_simple": matmul_minifloat_simple,
        "minifloat_ieee": matmul_minifloat_ieee,
        "msfp": matmul_msfp,
    },
    torch.matmul: {
        "integer": matmul_integer,
        "minifloat_simple": matmul_minifloat_simple,
        "minifloat_ieee": matmul_minifloat_ieee,
        "msfp": matmul_msfp,
    },
    torch.bmm: {
        "integer": bmm_integer,
        "minifloat_simple": bmm_minifloat_simple,
        "minifloat_ieee": bmm_minifloat_ieee,
        "msfp": bmm_msfp,
    },
}

METHOD_MAP_NEO = {
    "add": {
        "integer": add_integer,
        "minifloat_simple": add_minifloat_simple,
        "minifloat_ieee": add_minifloat_ieee,
        "msfp": add_msfp,
    },
    "relu": {
        "integer": relu_integer,
        "minifloat_simple": relu_minifloat_simple,
        "minifloat_ieee": relu_minifloat_ieee,
        "msfp": relu_msfp,
    },
    "matmul": {
        "integer": matmul_integer,
        "minifloat_simple": matmul_minifloat_simple,
        "minifloat_ieee": matmul_minifloat_ieee,
        "msfp": matmul_msfp,
    },
    "bmm": {
        "integer": bmm_integer,
        "minifloat_simple": bmm_minifloat_simple,
        "minifloat_ieee": bmm_minifloat_ieee,
        "msfp": bmm_msfp,
    },
}
