import logging

import torch
import torch.nn as nn
from torch import Tensor

from ..quantizers.integer import integer_floor_quantizer
from .fixed_signed_cast import _fixed_signed_cast_model


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _fixed_rms_norm_2d_model(
    x: Tensor,
    acc_width: int,
    acc_frac_width: int,
    inv_sqrt_width: int,
    inv_sqrt_frac_width: int,
    out_width: int,
    out_frac_width: int,
):
    logger.debug("Input:")
    logger.debug(x[0])

    # Sum of Squares
    sum_sq = torch.square(x).sum(dim=(1, 2, 3), keepdim=True)
    sum_sq = integer_floor_quantizer(sum_sq, acc_width, acc_frac_width)
    logger.debug("Sum of Squares:")
    logger.debug(sum_sq[0])

    # Divide to get mean square
    mean_sq = sum_sq / (x.shape[1] * x.shape[2] * x.shape[3])
    mean_sq = integer_floor_quantizer(mean_sq, acc_width, acc_frac_width)
    logger.debug("Mean Square:")
    logger.debug(mean_sq[0])

    # Get inverse sqrt of mean square
    # inv_sqrt = inv_sqrt_model(mean_sq)  # TODO: Add inv sqrt model
    inv_sqrt = torch.full_like(mean_sq, 0.25)  # TODO: remove this later
    inv_sqrt = integer_floor_quantizer(inv_sqrt, inv_sqrt_width, inv_sqrt_frac_width)
    logger.debug("Inverse SQRT:")
    logger.debug(inv_sqrt[0])

    # Norm calculation
    norm_out = x * inv_sqrt
    logger.debug("Norm:")
    logger.debug(norm_out[0])
    norm_out_float, norm_int_out = _fixed_signed_cast_model(
        norm_out, out_width, out_frac_width,
        symmetric=False, rounding_mode="floor"
    )

    return norm_out_float, norm_int_out


class RMSNorm2dInteger(nn.Module):
    def __init__(
        self,
        acc_width: int,
        acc_frac_width: int,
        inv_sqrt_width: int,
        inv_sqrt_frac_width: int,
        out_width: int,
        out_frac_width: int,
    ) -> None:
        super().__init__()

        self.acc_width = acc_width
        self.acc_frac_width = acc_frac_width
        self.inv_sqrt_width = inv_sqrt_width
        self.inv_sqrt_frac_width = inv_sqrt_frac_width
        self.out_width = out_width
        self.out_frac_width = out_frac_width

    def forward(self, x: Tensor):
        return _fixed_rms_norm_2d_model(
            x=x,
            acc_width=self.acc_width,
            acc_frac_width=self.acc_frac_width,
            inv_sqrt_width=self.inv_sqrt_width,
            inv_sqrt_frac_width=self.inv_sqrt_frac_width,
            out_width=self.out_width,
            out_frac_width=self.out_frac_width,
        )
