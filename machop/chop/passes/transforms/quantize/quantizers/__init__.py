from .block_fp import block_fp_quantizer
from .block_log import block_log_quantizer
from .block_minifloat import block_minifloat_quantizer
from .integer import integer_quantizer
from .log import log_quantizer
from .minifloat import minifloat_denorm_quantizer, minifloat_ieee_quantizer

quantizer_map = {
    "log": log_quantizer,
    "block_log": block_log_quantizer,
    "minifloat_denorm": minifloat_denorm_quantizer,
    "minifloat_ieee": minifloat_ieee_quantizer,
    "block_minifloat": block_minifloat_quantizer,
    "block_fp": block_fp_quantizer,
    "integer": integer_quantizer,
}

# import operator
# from typing import Dict, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .functions import (
#     add_block_minifloat,
#     add_integer,
#     add_log,
#     add_minifloat_ieee,
#     add_minifloat_denorm,
#     add_msfp,
#     bmm_block_minifloat,
#     bmm_integer,
#     bmm_log,
#     bmm_minifloat_ieee,
#     bmm_minifloat_denorm,
#     bmm_msfp,
#     matmul_block_minifloat,
#     matmul_integer,
#     matmul_log,
#     matmul_minifloat_ieee,
#     matmul_minifloat_denorm,
#     matmul_msfp,
#     relu_block_minifloat,
#     relu_integer,
#     relu_log,
#     relu_minifloat_ieee,
#     relu_minifloat_denorm,
#     relu_msfp,
# )
# from .layers import AvgPool2dInteger  # AddInteger,
# from .layers import (
#     AdaptiveAvgPool2dInteger,
#     Conv1dBlockMinifloat,
#     Conv1dInteger,
#     Conv1dLog,
#     Conv1dMinifloatIEEE,
#     Conv1dMinifloatDenorm,
#     Conv1dMSFP,
#     Conv2dBlockMinifloat,
#     Conv2dInteger,
#     Conv2dLog,
#     Conv2dMinifloatIEEE,
#     Conv2dMinifloatDenorm,
#     Conv2dMSFP,
#     LinearBlockMinifloat,
#     LinearInteger,
#     LinearLog,
#     LinearMinifloatIEEE,
#     LinearMinifloatDenorm,
#     LinearMSFP,
#     ReLUBlockMinifloat,
#     ReLUInteger,
#     ReLULog,
#     ReLUMinifloatIEEE,
#     ReLUMinifloatDenorm,
#     ReLUMSFP,
# )

# MODULE_CLS_MAP = {
#     nn.Linear: {
#         "integer": LinearInteger,
#         "minifloat_denorm": LinearMinifloatDenorm,
#         "minifloat_ieee": LinearMinifloatIEEE,
#         "log": LinearLog,
#         "msfp": LinearMSFP,
#         "block_minifloat": LinearBlockMinifloat,
#     },
#     nn.ReLU: {
#         "integer": ReLUInteger,
#         "minifloat_denorm": ReLUMinifloatDenorm,
#         "minifloat_ieee": ReLUMinifloatIEEE,
#         "log": ReLULog,
#         "msfp": ReLUMSFP,
#         "block_minifloat": ReLUBlockMinifloat,
#     },
#     nn.Conv1d: {
#         "integer": Conv1dInteger,
#         "minifloat_denorm": Conv1dMinifloatDenorm,
#         "minifloat_ieee": Conv1dMinifloatIEEE,
#         "log": Conv1dLog,
#         "msfp": Conv1dMSFP,
#         "block_minifloat": Conv1dBlockMinifloat,
#     },
#     nn.Conv2d: {
#         "integer": Conv2dInteger,
#         "minifloat_denorm": Conv2dMinifloatDenorm,
#         "minifloat_ieee": Conv2dMinifloatIEEE,
#         "log": Conv2dLog,
#         "msfp": Conv2dMSFP,
#         "block_minifloat": Conv2dBlockMinifloat,
#     },
#     nn.AvgPool2d: {
#         "integer": AvgPool2dInteger,
#     },
#     nn.AdaptiveAvgPool2d: {"integer": AdaptiveAvgPool2dInteger},
# }

# QUANTIZED_MODULE_CLASSES = []
# for k, v in MODULE_CLS_MAP.items():
#     for kk, vv in v.items():
#         QUANTIZED_MODULE_CLASSES.append(vv)

# FUNC_MAP = {
#     operator.add: {
#         "integer": add_integer,
#         "minifloat_denorm": add_minifloat_denorm,
#         "minifloat_ieee": add_minifloat_ieee,
#         "log": add_log,
#         "msfp": add_msfp,
#         "block_minifloat": add_block_minifloat,
#     },
#     torch.add: {
#         "integer": add_integer,
#         "minifloat_denorm": add_minifloat_denorm,
#         "minifloat_ieee": add_minifloat_ieee,
#         "log": add_log,
#         "msfp": add_msfp,
#         "block_minifloat": add_block_minifloat,
#     },
#     F.relu: {
#         "integer": relu_integer,
#         "minifloat_denorm": relu_minifloat_denorm,
#         "minifloat_ieee": relu_minifloat_ieee,
#         "log": relu_log,
#         "msfp": relu_msfp,
#         "block_minifloat": relu_block_minifloat,
#     },
#     operator.matmul: {
#         "integer": matmul_integer,
#         "minifloat_denorm": matmul_minifloat_denorm,
#         "minifloat_ieee": matmul_minifloat_ieee,
#         "log": matmul_log,
#         "msfp": matmul_msfp,
#         "block_minifloat": matmul_block_minifloat,
#     },
#     torch.matmul: {
#         "integer": matmul_integer,
#         "minifloat_denorm": matmul_minifloat_denorm,
#         "minifloat_ieee": matmul_minifloat_ieee,
#         "log": matmul_log,
#         "msfp": matmul_msfp,
#         "block_minifloat": matmul_block_minifloat,
#     },
#     torch.bmm: {
#         "integer": bmm_integer,
#         "minifloat_denorm": bmm_minifloat_denorm,
#         "minifloat_ieee": bmm_minifloat_ieee,
#         "log": bmm_log,
#         "msfp": bmm_msfp,
#         "block_minifloat": bmm_block_minifloat,
#     },
# }

# QUANTIZED_FUNC_CLASSES = []
# for k, v in FUNC_MAP.items():
#     for kk, vv in v.items():
#         QUANTIZED_FUNC_CLASSES.append(vv)

# METHOD_MAP = {
#     "add": {
#         "integer": add_integer,
#         "minifloat_denorm": add_minifloat_denorm,
#         "minifloat_ieee": add_minifloat_ieee,
#         "log": add_log,
#         "msfp": add_msfp,
#         "block_minifloat": add_block_minifloat,
#     },
#     "relu": {
#         "integer": relu_integer,
#         "minifloat_denorm": relu_minifloat_denorm,
#         "minifloat_ieee": relu_minifloat_ieee,
#         "log": relu_log,
#         "msfp": relu_msfp,
#         "block_minifloat": relu_block_minifloat,
#     },
#     "matmul": {
#         "integer": matmul_integer,
#         "minifloat_denorm": matmul_minifloat_denorm,
#         "minifloat_ieee": matmul_minifloat_ieee,
#         "log": matmul_log,
#         "msfp": matmul_msfp,
#         "block_minifloat": matmul_block_minifloat,
#     },
#     "bmm": {
#         "integer": bmm_integer,
#         "minifloat_denorm": bmm_minifloat_denorm,
#         "minifloat_ieee": bmm_minifloat_ieee,
#         "log": bmm_log,
#         "msfp": bmm_msfp,
#         "block_minifloat": bmm_block_minifloat,
#     },
# }
