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
