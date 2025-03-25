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

def gelu_relu6_approx(x, q_config):
    """
    GELU Approximation:
    GELU(x) ≈ x * ReLU6(1.702 * x + 3) / 6
    """
    qx, mx, ex = mxint_hardware(
        x,
        {
            "width": q_config["data_in_width"],
            "exponent_width": q_config["data_in_exponent_width"],
            "round_bits": 4,
        }, 
        parallelism=q_config["data_in_parallelism"]
    )
    qout = x * torch.nn.functional.relu6(1.702*x + 3)/6
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

def gelu_poly2_approx(x, q_config, alpha=0.2888, beta=1.769):
    """
    GELU Approximation:
    GELU(x) ≈ x * 0.5 * (1 + L(x / sqrt(2)))
    where:
    L(x) = sign(x) * [ alpha * (clip(|x|, max=-beta) + beta)^2 + 1 ]
    """
    qx, mx, ex = mxint_hardware(
        x,
        {
            "width": q_config["data_in_width"],
            "exponent_width": q_config["data_in_exponent_width"],
            "round_bits": 4,
        }, 
        parallelism=q_config["data_in_parallelism"]
    )
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))
    x_div = x / sqrt2
    abs_x = torch.abs(x_div)
    clipped = torch.clamp(abs_x, max=-beta) + beta
    poly = alpha * (clipped ** 2) + 1
    L = torch.sign(x_div) * poly
    qout = x * 0.5 * (1 + L)
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

    if q_config.get("enable_internal_width"):
        #manage the shape
        original_shape = qx.shape
        t1, t0 = mx.shape[-2:]
        p1, p0 = q_config["data_in_parallelism"]
        qx = reshape_to_block(qx, t1,t0, p1, p0)
        mx = reshape_to_block(mx, t1, t0, p1, p0)
        ex = ex.unsqueeze(-1).unsqueeze(-1)

        # set the hash config
        hash_in_width = q_config["hash_in_int_width"] + q_config["hash_in_frac_width"]
        hash_in_frac_width = q_config["hash_in_frac_width"]
        upper_bound = (2**(hash_in_width - 1) - 1) / 2**hash_in_frac_width
        lower_bound = -(2**(hash_in_width - 1)) / 2**hash_in_frac_width

        hash_out_width = q_config["hash_out_int_width"] + q_config["hash_out_frac_width"]
        hash_out_frac_width = q_config["hash_out_frac_width"]
        qx = _integer_quantize(qx, hash_in_width, hash_in_frac_width)
        qout = torch.relu(qx)
        eout = ex
        remaining = (qx > lower_bound) & (qx < upper_bound)
        # hash loss
        qgelu = _integer_quantize(torch.nn.GELU()(qx), hash_out_width, hash_out_frac_width)
        mgelu = qgelu * 2**(hash_out_frac_width - 1) // 2**ex
        qgelu = mgelu * 2**ex / 2**(hash_out_frac_width - 1)

        qout[remaining] = qgelu[remaining]
        qout = reshape_back(qout, t1, t0, p1, p0)
        qout = qout.reshape(original_shape)
    else:
        qout = torch.nn.GELU()(qx)

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
        # out, _, _  = mxint_gelu(x, self.q_config)
        # out, _, _ = gelu_relu6_approx(x, self.q_config)
        out, _, _ = gelu_poly2_approx(x, self.q_config)
        return out

