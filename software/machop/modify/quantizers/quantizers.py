import torch
from .utils import my_clamp, my_round
from typing import Union
from torch import Tensor
from numpy import ndarray


def integer_quantizer(
        x: Union[Tensor, ndarray], 
        bits: int, 
        bias: int):
    """Do linear quantization to input according to a scale and number of bits"""
    thresh =  2 ** (bits - 1)
    scale = 2 ** bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh-1).div(scale)


def block_quantizer(
        x: Union[Tensor, ndarray],
        bits: int,
        bias: int,
        block_size: int = 1):
    """Do linear quantization to input according to a scale and number of bits"""
    thresh =  2 ** (bits - 1)
    scale = 2 ** bias
    return my_clamp(my_round(x.mul(scale).div(block_size)), -thresh, thresh-1).mul(block_size).div(scale
)

