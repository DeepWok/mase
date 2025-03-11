# models.py
import torch
import torch.nn as nn
import math
from typing import List, Union, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Literal, Optional, Tuple, Union, Dict
from enum import Enum
from functools import partial
from tqdm import tqdm
from chop.nn.quantizers.integer import _integer_quantize
from .quantizers import mxint_hardware
from .utils import reshape_to_block, reshape_back

def mxint_gelu(x, q_config):
    """Vectorized range reduction"""
    qx, mx, ex = mxint_hardware(
        x,
        {
            "width": q_config["data_in_width"],
            "exponent_width": q_config["data_in_exponent_width"],
            "round_bits": 4,
        }, 
        parallelism=q_config["data_in_parallelism"]
    )
    # first

    original_shape = qx.shape
    t1, t0 = mx.shape[-2:]
    p1, p0 = q_config["data_in_parallelism"]
    qx = reshape_to_block(qx, t1,t0, p1, p0)
    mx = reshape_to_block(mx, t1, t0, p1, p0)
    ex = ex.unsqueeze(-1).unsqueeze(-1)

    qout = torch.relu(qx)
    eout = ex
    remaining = (qx > -3) & (qx < 3)

    # data_width_loss
    # avoid quant_loss here
    # we will need to shift it to 
    # in hardware qx is lossless
    VALID_WIDTH = q_config["data_in_width"] + 2
    HASH_OUT_WIDTH = q_config["hash_out_width"]
    HASH_OUT_FRAC_WIDTH = HASH_OUT_WIDTH - 3
    # hash loss
    qgelu = _integer_quantize(torch.nn.GELU()(qx), HASH_OUT_WIDTH, HASH_OUT_FRAC_WIDTH)
    mgelu = qgelu * 2**(HASH_OUT_WIDTH - 1) // 2**ex
    qgelu = mgelu * 2**ex / 2**(HASH_OUT_WIDTH - 1)

    qout[remaining] = qgelu[remaining]
    qout = reshape_back(qout, t1, t0, p1, p0)
    qout = qout.reshape(original_shape)
    qx, mx, ex = mxint_hardware(
        qout,
        {
            "width": q_config["data_out_width"],
            "exponent_width": q_config["data_out_exponent_width"],
            "round_bits": 4,
        }, 
        parallelism=q_config["data_out_parallelism"]
    )
    return qx, mx, ex

class MXIntGELU(nn.Module):
    def __init__(self, q_config: Dict = {}):
        super().__init__()
        self.q_config = q_config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _  = mxint_gelu(x, self.q_config)
        return out

