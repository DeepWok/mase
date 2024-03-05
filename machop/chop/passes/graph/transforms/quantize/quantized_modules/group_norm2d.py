import logging

import torch
import torch.nn as nn
from torch import Tensor

from ..quantizers.integer import integer_floor_quantizer
from .fixed_signed_cast import _fixed_signed_cast_model


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _fixed_group_norm_2d_model(
    x: Tensor,
    in_width: int,
    in_frac_width: int,
    variance_width: int,
    variance_frac_width: int,
    inv_sqrt_width: int,
    inv_sqrt_frac_width: int,
    out_width: int,
    out_frac_width: int,
):
    logger.debug("Input:")
    logger.debug(x[0])

    # Mean calculation
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    logger.debug("Mu:")
    logger.debug(mu[0])
    mu = integer_floor_quantizer(mu, in_width, in_frac_width)
    logger.debug("Mu:")
    logger.debug(mu[0])

    # Variance calculation
    var = ((x - mu) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    var = integer_floor_quantizer(var, variance_width, variance_frac_width)
    logger.debug("Variance:")
    logger.debug(var[0])

    # Inverse Square Root calculation
    # inv_sqrt = inv_sqrt_model(var)  # TODO: Add inv sqrt model
    inv_sqrt = torch.full_like(var, 0.25)  # TODO: remove this later
    inv_sqrt = integer_floor_quantizer(inv_sqrt, inv_sqrt_width, inv_sqrt_frac_width)
    logger.debug("Inverse SQRT:")
    logger.debug(inv_sqrt[0])

    # Norm calculation
    diff = x - mu
    logger.debug("Diff:")
    logger.debug(diff[0])
    norm_out = diff * inv_sqrt
    logger.debug("Norm:")
    logger.debug(norm_out[0])
    norm_out_float, norm_int_out = _fixed_signed_cast_model(
        norm_out, out_width, out_frac_width,
        symmetric=False, rounding_mode="floor"
    )
    logger.debug("Norm (Casted):")
    logger.debug(norm_out_float[0])

    logger.debug("Norm (unsigned):")
    logger.debug(norm_int_out[0])

    return norm_out_float, norm_int_out


class GroupNorm2dInteger(nn.Module):
    def __init__(
        self,
        in_width: int,
        in_frac_width: int,
        variance_width: int,
        variance_frac_width: int,
        inv_sqrt_width: int,
        inv_sqrt_frac_width: int,
        out_width: int,
        out_frac_width: int,
    ) -> None:
        super().__init__()

        self.in_width = in_width
        self.in_frac_width = in_frac_width
        self.variance_width = variance_width
        self.variance_frac_width = variance_frac_width
        self.inv_sqrt_width = inv_sqrt_width
        self.inv_sqrt_frac_width = inv_sqrt_frac_width
        self.out_width = out_width
        self.out_frac_width = out_frac_width

    def forward(self, x: Tensor):
        return _fixed_group_norm_2d_model(
            x=x,
            in_width=self.in_width,
            in_frac_width=self.in_frac_width,
            variance_width=self.variance_width,
            variance_frac_width=self.variance_frac_width,
            inv_sqrt_width=self.inv_sqrt_width,
            inv_sqrt_frac_width=self.inv_sqrt_frac_width,
            out_width=self.out_width,
            out_frac_width=self.out_frac_width,
        )
