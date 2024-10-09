import torch
from functools import partial
import torch.nn.functional as F
from torch import Tensor


def mxint_quant_block(
    x, width: int = 12, exponent_width: int = 6, exponent: int = None
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

    # exponent
    if exponent == None:
        exponent = torch.ceil(torch.log2(x.abs().max())) - exponent_bias
        exponent = torch.clamp(exponent, exponent_min, exponent_max)
    # mantissa
    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
    mantissa = x / 2**exponent
    mantissa = torch.clamp(mantissa.floor(), int_min, int_max)
    q_x = (2**exponent) * mantissa
    return q_x


def mxint_hardware(tensor, q_config, parallelism):
    """
    - For hardware efficiency, the block will be set based on parallelism
    - This will reshape all the input to a 3D matrix (other dimension will be packed into the first dimension)
    - Then will quantize every block of the 2D matrix in the reshaped input tensor.
    - The block size will be [parallelism[0], parallelism[1]]
    ---
    - q_config: assume to be a dict, for example
    {
        "width": 8,
        "exponent_width": 4,
    }
    - parallelism: assume to be [tensor.shape[-2],tensor.shape[-1]]
    - `exponent_width`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used
    """
    original_shape = tensor.shape
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    if len(parallelism) == 1:
        parallelism = [1, parallelism[0]]

    p1 = parallelism[0]
    p0 = parallelism[1]
    t1 = tensor.shape[-2]
    t0 = tensor.shape[-1]
    assert (
        t1 % p1 == 0 and t0 % p0 == 0
    ), f"""The Block should be able to completely segment the tensor size, 
    t1 = {t1}, p1 = {p1}, t0 = {t0}, p0 = {p0}"""
    reshaped_tensor = tensor.reshape(-1, t1 // p1, p1, t0 // p0, p0).permute(
        0, 1, 3, 2, 4
    )

    # Quantize
    quantizer = partial(mxint_quant_block, **q_config)
    reshaped_tensor = torch.tensor(reshaped_tensor.reshape(-1, p1 * p0))
    for i in range(reshaped_tensor.shape[0]):
        reshaped_tensor[i] = quantizer(reshaped_tensor[i])
    qtensor = (
        reshaped_tensor.reshape(-1, t1 // p1, t0 // p0, p1, p0)
        .permute(0, 1, 3, 2, 4)
        .reshape(original_shape)
    )
    return qtensor
