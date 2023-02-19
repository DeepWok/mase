import torch
from .utils import my_clamp, my_round
from typing import Union
from torch import Tensor
from numpy import ndarray


def integer_quantizer(x: Union[Tensor, ndarray], bits: int, bias: int):
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2**(bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


def minifloat_quantizer(x: Union[Tensor,
                                 ndarray], bits: int, exponent_bits: int,
                        exponent_bias: Union[int, Tensor, ndarray, None]):
    # we will work out the mantissa bits
    # 1 sign bit, exponent, then mantissa
    mantissa_bits = bits - exponent_bits - 1

    # decompose the number, the smallest possible minifloat value
    sign = torch.sign(x)
    value = torch.abs(x)
    exponent = torch.floor(torch.log2(value))
    mantissa = value / 2**exponent

    if exponent_bias is None:
        # ensure no overflow
        exponent_max = torch.ceil(torch.log2(x.max()))
        exponent_bias = 2**exponent_bits - 1 - exponent_max
        exponent_min = -exponent_bias
    else:
        exponent_min = -exponent_bias
        exponent_max = 2**exponent_bits - 1 - exponent_bias

    exponent = my_clamp(exponent, exponent_min, exponent_max)

    shift = 2**mantissa_bits
    mantissa = my_round(mantissa * shift) / shift
    return sign * mantissa * (2**exponent)


def log_quantizer(x: Union[Tensor, ndarray], bits: int,
                  exponent_bias: Union[int, Tensor, ndarray, None]):
    raise NotImplementedError()


def block_and_padd(x: Tensor, block_size: int = 16):
    ''' 
    Pad zeros so that the size of the input is a multiple of the block_size 
    The output now should have an additional dimension of size block_size
    '''
    if len(x.shape) == 2:
        batch, size = x.shape
    if len(x.shape) == 3:
        batch, pd, size = x.shape

    num_buckets = ceil(size / block_size)
    new_size = num_buckets * block_size
    diff = new_size - size
    padded = F.pad(x, (0, diff))

    # reshape to be a multiple of the bucket size, so that we can quantize it in chunks
    if len(x.shape) == 2:
        padded = padded.reshape(batch, num_buckets, block_size)
    elif len(x.shape) == 3:
        padded = padded.reshape(batch, pd, num_buckets, block_size)

    if len(x.shape) == 2:
        per_block_max = padded.abs().max(dim=2)[0]
    elif len(x.shape) == 3:
        per_block_max = padded.abs().max(dim=3)[0]
    else:
        raise ValueError(f'{x.shape} size for the input is not supported!')

    return padded, per_block_max


def block_fp_quantizer(x: Union[Tensor, ndarray],
                       bits: int,
                       block_size: int = 16):
    # WARNING: needs testing
    shape = None
    if len(x.shape) == 2:
        batch, size = x.shape
        shape = 'dim2'
    if len(x.shape) == 3:
        import pdb
        pdb.set_trace()
        batch, dim_a, size = x.shape
        shape = 'dim3'

    x, per_block_max = block_and_padd(x, block_size=block_size)
    # fill zeros
    per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    scale = torch.ceil(torch.log2(per_block_max))
    scale = 2**(bits - scale - 1)
    scale = scale.unsqueeze(-1)
    max_value = (2**(scale - 1))

    quantized = my_clamp(my_round(x.mul(scale)), -max_value,
                         max_value - 1).div(scale)
    if shape == 'dim2':
        quantized = quantized.reshape(batch, -1)[:, :size]
    if shape == 'dim3':
        quantized = quantized.reshape(batch, dim_a, -1)[:, :, :size]
    return quantized


def block_minifloat_quantizer(x: Union[Tensor, ndarray],
                              bits: int,
                              exponent_bits: int,
                              block_size: int = 16):
    # WARNING:  needs testing
    if len(x.shape) == 2:
        batch, size = x.shape
    if len(x.shape) == 3:
        batch, dim_a, size = x.shape
    x, per_block_max = block_and_padd(x, block_size=block_size)
    # fill zeros
    per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()

    bias = torch.ceil(torch.log2(per_block_max))
    quantized = minifloat_quantizer(x, bits, exponent_bits, bias)

    if len(input.shape) == 2:
        quantized = quantized.reshape(batch, -1)[:, :size]
    if len(input.shape) == 3:
        quantized = quantized.reshape(batch, dim_a, -1)[:, :, :size]
    return quantized


def block_log_quantizer(x: Union[Tensor, ndarray],
                        bits: int,
                        block_size: int = 16):
    raise NotImplementedError()
