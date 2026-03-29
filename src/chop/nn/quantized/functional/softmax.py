from functools import partial

import torch
from torch import Tensor

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim


def softmax_mxfp(x: Tensor, config: dict = None, dim: int = -1) -> Tensor:
    x_block_size = config["data_in_block_size"]
    x_exp_bits = config["data_in_exponent_width"]
    x_frac_bits = config["data_in_frac_width"]

    x_quantizer = partial(
        mxfp_quantizer,
        block_size=x_block_size,
        element_exp_bits=x_exp_bits,
        element_frac_bits=x_frac_bits,
        block_dim=-1,
    )

    x = x_quantizer(x)
    return torch.nn.functional.softmax(x.to(torch.float32), dim=dim).to(x.dtype)


def softmax_mxint(x: Tensor, config: dict = None, dim: int = -1) -> Tensor:
    x_block_size = config["data_in_block_size"]
    x_element_bits = config["data_in_width"]

    x_quantizer = partial(
        mxint_quantizer,
        block_size=x_block_size,
        element_bits=x_element_bits,
        block_dim=-1,
    )

    x = x_quantizer(x)
    return torch.nn.functional.softmax(x.to(torch.float32), dim=dim).to(x.dtype)


def softmax_minifloat(x: Tensor, config: dict = None, dim: int = -1) -> Tensor:
    x_exp_bits = config["data_in_exponent_width"]
    x_frac_bits = config["data_in_frac_width"]

    x_quantizer = partial(
        minifloat_quantizer_sim,
        minifloat_meta=MinifloatMeta(
            exp_bits=x_exp_bits,
            frac_bits=x_frac_bits,
            is_finite=config.get("data_in_is_finite", True),
            round_mode=config.get("data_in_round_mode", "rn"),
        ),
    )

    x = x_quantizer(x)
    return torch.nn.functional.softmax(x.to(torch.float32), dim=dim).to(x.dtype)
