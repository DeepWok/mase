import torch
from functools import partial
import torch.nn.functional as F
from torch import Tensor
from chop.nn.quantized.modules.linear import _LinearBase
from .utils import reshape_to_block, reshape_back


def mxint_quant_block(
    x, width: int = 12, exponent_width: int = 6, exponent: int = None, round_bits: int = 4,
):
    """
    - Idea from https://arxiv.org/pdf/2310.10537
    - Convert IEEE FP32/64 to Integer with sharing scale
    - The main difference between is the sharing scale do not support NAN representation
    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used

    """
    exponent_bias = 2 ** (exponent_width - 1)
    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias
    
    # Vectorized max and log2 operations
    abs_max = x.abs().max(dim=-1, keepdim=True).values
    # abs_max = x.abs().max()
    log2 = torch.log2(abs_max + torch.finfo(torch.float32).tiny)

    exponent = torch.ceil(log2) 
    exponent[exponent == log2] += 1
    exponent = torch.clamp(exponent, exponent_min, exponent_max)
    
    # Vectorized mantissa calculation
    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
    mantissa = x * (2 ** (width - 1)) / 2**exponent
    mantissa = mantissa * 2 ** round_bits
    mantissa = torch.floor(mantissa)
    mantissa = mantissa / 2 ** round_bits
    mantissa = torch.round(mantissa)
    mantissa = torch.clamp(mantissa, int_min, int_max)
    q_x = (2**exponent) * mantissa /(2 ** (width - 1))  
    return q_x, mantissa, exponent

def mxint_hardware(tensor, q_config, parallelism):
    """
    mxint hardware-aware quantization implementation
    - `q_config`: the quantization configuration
    - `parallelism`: the parallelism of the tensor
    - return: the quantized tensor, mantissa, and exponent
    """

    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    if len(parallelism) == 1:
        parallelism = [1, parallelism[0]]

    p1, p0 = parallelism
    t1, t0 = tensor.shape[-2:]
    
    original_mshape = tensor.shape
    original_eshape = torch.Size([t1//p1, t0//p0]) if len(tensor.shape) <=2 else torch.Size([*tensor.shape[:-2],t1//p1, t0//p0])
    assert (t1 % p1 == 0 and t0 % p0 == 0), \
        f"Block size mismatch: t1={t1}, p1={p1}, t0={t0}, p0={p0}"
    
    # Single reshape and permute operation
    block_tensor = reshape_to_block(tensor, t1, t0, p1, p0).reshape(-1, p1*p0)
    qtensor, mantissa, exponent = mxint_quant_block(block_tensor, **q_config)
    
    qtensor = reshape_back(qtensor, t1, t0, p1, p0)
    mantissa = reshape_back(mantissa, t1, t0, p1, p0)
    qtensor = qtensor.reshape(original_mshape)
    mantissa = mantissa.reshape(original_mshape)
    exponent = exponent.reshape(original_eshape)
    # Efficient shape restoration
    return qtensor, mantissa, exponent