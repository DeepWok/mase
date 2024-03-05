import torch
import torch.nn as nn
from torch import Tensor

from mase_cocotb.utils import signed_to_unsigned
from ..quantizers.utils import my_clamp, my_floor, my_round


def _fixed_signed_cast_model(
    float_input, out_width, out_frac_width, symmetric, rounding_mode
):
    scaled_float = float_input * (2 ** out_frac_width)
    if rounding_mode == "floor":
        out_int = my_floor(scaled_float)
    elif rounding_mode == "round_nearest_half_even":
        out_int = my_round(scaled_float)
    else:
        raise Exception("Rounding mode not recognised.")
    out_int = my_clamp(
        out_int,
        -(2**(out_width-1))+1 if symmetric else -(2**(out_width-1)),
        (2**(out_width-1))-1
    )
    out_float = out_int / (2 ** out_frac_width)
    # out_uint is a non-differentiable path
    out_uint = signed_to_unsigned(out_int.int(), out_width)
    return out_float, out_uint


class FixedSignedCastInteger(nn.Module):
    def __init__(
        self,
        out_width: int,
        out_frac_width: int,
        symmetric: bool = False,
        rounding_mode: str = "floor",
    ) -> None:
        super().__init__()

        self.out_width = out_width
        self.out_frac_width = out_frac_width
        self.symmetric = symmetric
        self.rounding_mode = rounding_mode

    def forward(self, x: Tensor):
        return _fixed_signed_cast_model(
            x, self.out_width, self.out_frac_width, self.symmetric, self.rounding_mode
        )
