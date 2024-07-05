import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# from .quantizers import integer_quantizer
from .utils import block, my_clamp, my_round, unblock, my_floor


def integer_quantizer_for_hw(x: Tensor, width: int, frac_width: int):
    thresh = 2 ** (width - 1)
    scale = 2**frac_width

    fixed_point_value = my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1)
    fixed_point_value = fixed_point_value.to(torch.int)
    fixed_point_value = fixed_point_value % (2**width)
    return fixed_point_value


def unsigned_integer_quantizer_for_hw(x: Tensor, width: int, frac_width: int):
    thresh = 2**width - 1
    scale = 2**frac_width

    fixed_point_value = my_clamp(my_floor(x.mul(scale)), 0, thresh)
    fixed_point_value = fixed_point_value.to(torch.int)
    fixed_point_value = fixed_point_value % (2**width)
    return fixed_point_value


def integer_floor_quantizer_for_hw(x: Tensor, width: int, frac_width: int):
    thresh = 2 ** (width - 1)
    scale = 2**frac_width

    fixed_point_value = my_clamp(my_floor(x.mul(scale)), -thresh, thresh - 1)
    fixed_point_value = fixed_point_value.to(torch.int)
    fixed_point_value = fixed_point_value % (2**width)
    return fixed_point_value


# sw_quantizer_to_hw_quantizer = {integer_quantizer: integer_quantizer_for_hw}
