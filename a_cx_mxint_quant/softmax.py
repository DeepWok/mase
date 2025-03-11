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
from .quantizers import mxint_quant_block, mxint_hardware
from chop.nn.quantizers.integer import integer_quantizer, integer_floor_quantizer
from functools import partial
from tqdm import tqdm

class MXIntHardwareExp(nn.Module):
    def __init__(self, q_config: Dict = {}):
        super().__init__()
        self.q_config = q_config

    def hardware_range_reduction(self, qx, data_r_width, data_n_width) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform range reduction: x = r + n*ln(2)
        Returns (r, n) where r is remainder and n is integer power
        """
        coefficient_quant_block = partial(
            mxint_quant_block, 
            width=8,
            exponent_width=4 
            )
        self.log2_e, _, _ = coefficient_quant_block(torch.log2(torch.tensor(math.e)))
        new_mx = qx * self.log2_e
        new_mx = integer_floor_quantizer(new_mx, data_n_width + data_r_width - 1, data_r_width - 1)
        n = new_mx.floor()
        r = new_mx - n
        return r, n

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        qx, mx, ex = mxint_hardware(x, 
                           {
                                 'width': self.q_config.get('data_in_width'),
                                 'exponent_width': self.q_config.get('data_in_exponent_width')
                           },
                           parallelism=[1,1])

        mr, n = self.hardware_range_reduction(qx, self.q_config.get('data_r_width'), self.q_config.get('data_out_exponent_width'))
        mexp = 2 ** mr 
        mexp = integer_quantizer(mexp, self.q_config.get('data_out_width'), self.q_config.get('data_out_width') - 2)
        mexp = mexp * 2 ** (self.q_config.get('data_out_width') - 2)
        eexp = n
        qexp = mexp * 2 ** eexp / 2 ** (self.q_config.get('data_out_width') - 2)
        
        return qexp, mexp, eexp

from tqdm import tqdm
# CX: Set a new search 
# accumulator depth should be in the first dimension
def mxint_accumulator(mx,ex):
    out = mx[0]
    emax = ex[0]
    for i in range(1, mx.shape[0]):
        old_max = emax
        emax = torch.max(emax, ex[i])
        in_out = out // 2**(emax - old_max)
        in_mx = mx[i]// 2**(emax - ex[i])
        out = in_out + in_mx 
        # breakpoint()
    return out, emax


class MXIntSoftmax(nn.Module):
    def __init__(self,q_config: Dict = {}):
        super().__init__()
        self.q_config = q_config
        self.exp_module = MXIntHardwareExp(q_config=q_config)

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        def exp(self, x):
            qexp, mexp, eexp = self.exp_module(x)
            return qexp, mexp, eexp

        def exp_sum(self, qexp, mexp, eexp):
            exp_sum_underflow_bits = self.q_config["exp_sum_underflow_bits"]
            mexp = (mexp) * 2**exp_sum_underflow_bits
            
            mexp= mexp.transpose(1,0)
            eexp = eexp.transpose(1,0)
            mexp_sum, eexp_sum = mxint_accumulator(mexp, eexp)
            qexp_sum = mexp_sum * 2**eexp_sum / 2**exp_sum_underflow_bits
            return qexp_sum, mexp_sum, eexp_sum
        
        def division(self, qexp, mexp, eexp, qexp_sum, mexp_sum, eexp_sum):
            division_underflow_bits = self.q_config["division_underflow_bits"]
            exp_sum_underflow_bits = self.q_config["exp_sum_underflow_bits"]
            mout = mexp * 2**(division_underflow_bits+exp_sum_underflow_bits) // mexp_sum
            eout = eexp - eexp_sum
            qout = mout * 2**eout / 2**division_underflow_bits

            qout, _, _ = mxint_hardware(
                qout,
                q_config = {
                    "width": self.q_config["data_width"],
                    "exponent_width": self.q_config["data_exponent_width"],
                },
                parallelism = [1,1]
            )

            return qout, mout, eout

        qexp, mexp, eexp = exp(self, x)
        qexp_sum, mexp_sum, eexp_sum = exp_sum(self, qexp, mexp, eexp)
        qout, mout, eout = division(self, qexp, mexp, eexp, qexp_sum, mexp_sum, eexp_sum)
        
        return qout