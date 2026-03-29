from functools import partial

from torch import Tensor

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer


def kv_cache_mxfp(
    key_states: Tensor,
    value_states: Tensor,
    config: dict = None,
) -> tuple[Tensor, Tensor]:
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

    return x_quantizer(key_states), x_quantizer(value_states)


def kv_cache_mxint(
    key_states: Tensor,
    value_states: Tensor,
    config: dict = None,
) -> tuple[Tensor, Tensor]:
    x_block_size = config["data_in_block_size"]
    x_element_bits = config["data_in_width"]

    x_quantizer = partial(
        mxint_quantizer,
        block_size=x_block_size,
        element_bits=x_element_bits,
        block_dim=-1,
    )

    return x_quantizer(key_states), x_quantizer(value_states)
