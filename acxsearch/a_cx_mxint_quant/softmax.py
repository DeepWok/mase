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

from .int_quant.quant_modules import IntSoftmaxImpl

class MXIntHardwareExp(nn.Module):
    def __init__(self, q_config: Dict = {}):
        super().__init__()
        self.q_config = q_config

    def hardware_range_reduction(self, qx, exp_width, exp_exponent_width) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform range reduction: x = r + n*ln(2)
        Returns (r, n) where r is remainder and n is integer power
        """
        coefficient_quant_block = partial(
            mxint_quant_block, 
            width=8,
            exponent_width=8 
            )
        self.log2_e, _, _ = coefficient_quant_block(torch.log2(torch.tensor(math.e)))
        new_mx = qx * self.log2_e
        new_mx = integer_floor_quantizer(new_mx, exp_width + exp_exponent_width - 1, exp_width - 1)
        n = new_mx.floor()
        r = new_mx - n
        return r, n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mr, n = self.hardware_range_reduction(x, self.q_config.get('exp_width'), self.q_config.get('exp_exponent_width'))
        mexp = 2 ** mr 
        mexp = integer_quantizer(mexp, self.q_config.get('exp_width'), self.q_config.get('exp_width') - 2)
        mexp = mexp * 2 ** (self.q_config.get('exp_width') - 2)
        eexp = n
        qexp = mexp * 2 ** eexp / 2 ** (self.q_config.get('exp_width') - 2)
        
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

        qx, _, _ = mxint_hardware(x, 
                           {
                                 'width': self.q_config.get('data_in_width'),
                                 'exponent_width': self.q_config.get('data_in_exponent_width')
                           },
                           parallelism=[1,1])
        
        if self.q_config.get("enable_internal_width"):
            if self.q_config.get("enable_mxint_exp"):
                qexp, mexp, eexp = exp(self, qx)
            else:
                qexp, mexp, eexp = mxint_hardware(
                    torch.exp(qx), 
                    {
                        "width": self.q_config.get('exp_width'),
                        "exponent_width": self.q_config.get('exp_exponent_width')
                    },
                    parallelism=[1,1]
                )
            
            if self.q_config.get("enable_mxint_exp_sum"):
                qexp_sum, mexp_sum, eexp_sum = exp_sum(self, qexp, mexp, eexp)
            else:
                qexp_sum, mexp_sum, eexp_sum = mxint_hardware(
                    torch.sum(qexp, dim=dim, keepdim=True), 
                    {
                        "width": self.q_config.get('exp_sum_width'),
                        "exponent_width": self.q_config.get('exp_sum_exponent_width')
                    },
                    parallelism=[1,1]
                )
            if self.q_config.get("enable_mxint_division"):
                qout, mout, eout = division(self, qexp, mexp, eexp, qexp_sum, mexp_sum, eexp_sum)
            else:
                qout, mout, eout = mxint_hardware(
                    qexp / qexp_sum, 
                    {
                        "width": self.q_config.get('data_out_width'),
                        "exponent_width": self.q_config.get('data_out_exponent_width')
                    },
                    parallelism=[1,1]
                )
        else:
            qout = torch.softmax(qx, dim=dim)
        
        qout, _, _ = mxint_hardware(
            qout,
            q_config = {
                "width": self.q_config["data_out_width"],
                "exponent_width": self.q_config["data_out_exponent_width"],
            },
            parallelism = [1,1]
        )
        return qout

from .int_quant.quant_modules import QuantAct
class IntSoftmax(nn.Module):
    def __init__(self, q_config: Dict = {}):
        super().__init__()
        self.q_config = q_config
        self.quant_act = QuantAct(
            activation_bit=self.q_config.get('in_width'), 
            act_range_momentum=self.q_config.get('in_range_momentum')
            )
        self.impl = IntSoftmaxImpl(output_bit=self.q_config.get('out_width'))
    
    def forward(self, x: torch.Tensor, dim) -> torch.Tensor:
        qx, scaling_factor = self.quant_act(x)
        qout, out_scaling_factor = self.impl(qx, scaling_factor)
        return qout

def approx_softmax(x, dim=-1):
    """
    Implements the softmax equation with approximations:
    Softmax(x_i) = exp{x_i - X_max - ln[∑exp(x_j - X_max)]}
    
    Using approximations:
    exp(x) = 2^((log2_e)*x) ≈ 2^u * (1 + 0.5v)
    ln(x) = ln(2) * log2(x) ≈ ln(2) * (w + k - 1)
    
    Parameters:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to apply softmax
        
    Returns:
        torch.Tensor: Softmax output with approximations
    """
    # Constants
    log2_e = math.log2(math.e)  # log_2(e)
    ln_2 = math.log(2)  # ln(2)
    
    # Find maximum for numerical stability
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_shifted = x - x_max
    
    # Approximate exp(x) = 2^((log2_e)*x) ≈ 2^u * (1 + 0.5v)
    def approx_exp(x):
        log2e_x = x * log2_e
        u = torch.floor(log2e_x)
        v = log2e_x - u
        return torch.pow(2.0, u) * (1.0 + 0.5 * v)
    
    # Approximate ln(x) = ln(2) * log2(x) ≈ ln(2) * (w + k - 1)
    def approx_ln(x):
        w = torch.floor(torch.log2(x))
        k = x / torch.pow(2.0, w)
        return ln_2 * (w + k - 1.0)
    
    # Apply approximations
    exp_x_shifted = approx_exp(x_shifted)
    exp_sum = torch.sum(exp_x_shifted, dim=dim, keepdim=True)
    log_sum = approx_ln(exp_sum)
    
    # Final softmax computation with approximation
    softmax_output = approx_exp(x_shifted - log_sum)
    
    return softmax_output

def power2n_softmax(x, dim=-1, n=1):
    """
    Implements the 2^n-SoftMax formula:
    
    X̂_i = X_i - max(X_i)
    2^n-SoftMax(X) = 2^(X̂_i) / ∑^J_j=1 2^(X̂_j)
    
    Parameters:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to apply softmax
        n (int): Power factor (default is 1 for regular 2-SoftMax)
        
    Returns:
        torch.Tensor: 2^n-SoftMax output
    """
    # Step 1: Calculate X̂_i = X_i - max(X_i)
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_hat = x - x_max
    
    # Step 2: Calculate 2^(n*X̂_i) for all i
    # For n=1, this is just 2^X̂_i
    power_x = torch.pow(2.0, n * x_hat)
    
    # Step 3: Calculate sum of 2^(n*X̂_j) across all j
    sum_power_x = torch.sum(power_x, dim=dim, keepdim=True)
    
    # Step 4: Calculate final result: 2^(n*X̂_i) / ∑ 2^(n*X̂_j)
    result = power_x / sum_power_x
    
    return result
# class MXIntSoftmax(nn.Module):
#     def __init__(self, q_config: Dict = {}):
#         super().__init__()
#         self.q_config = q_config

#     def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:

#         qx, _, _ = mxint_hardware(
#             x,
#             q_config = {
#                 "width": self.q_config["data_in_width"],
#                 "exponent_width": self.q_config["data_in_exponent_width"],
#             },
#             parallelism = [1,1]
#         )
#         qout = power2n_softmax(qx, dim)
#         qout, _, _ = mxint_hardware(
#             qout,
#             q_config = {
#                 "width": self.q_config["data_out_width"],
#                 "exponent_width": self.q_config["data_out_exponent_width"],
#             },
#             parallelism = [1,1]
#         )
#         return qout