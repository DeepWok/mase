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


def kv_cache_mxint_rotate(
    key_states: Tensor,
    value_states: Tensor,
    config: dict = None,
) -> tuple[Tensor, Tensor]:
    """KV-cache MXINT quantization with an exact Hadamard rotation around the
    quantizer (parallel to ``kv_cache_mxint``). ``hadamard_dim`` follows the
    head dim of K/V (last dim). Optional config keys:
    ``clip_search``, ``force_fp32_had``.
    """
    # Lazy import keeps fast_hadamard_transform optional at module load.
    from chop.nn.quantizers.rotation import mxint_rotate_quantizer

    x_block_size = config["data_in_block_size"]
    x_element_bits = config["data_in_width"]
    clip_search = config.get("clip_search", False)
    force_fp32_had = config.get("force_fp32_had", False)

    def x_quantizer(t: Tensor) -> Tensor:
        return mxint_rotate_quantizer(
            t,
            hadamard_dim=t.shape[-1],
            block_size=x_block_size,
            element_bits=x_element_bits,
            block_dim=-1,
            quantile_search=clip_search,
            force_fp32=force_fp32_had,
        )

    return x_quantizer(key_states), x_quantizer(value_states)


def kv_cache_mxfp_rotate(
    key_states: Tensor,
    value_states: Tensor,
    config: dict = None,
) -> tuple[Tensor, Tensor]:
    """KV-cache MXFP quantization with an exact Hadamard rotation around the
    quantizer (parallel to ``kv_cache_mxfp``). ``hadamard_dim`` follows the
    head dim of K/V (last dim). Optional config keys:
    ``clip_search``, ``force_fp32_had``.
    """
    # Lazy import keeps fast_hadamard_transform optional at module load.
    from chop.nn.quantizers.rotation import mxfp_rotate_quantizer

    x_block_size = config["data_in_block_size"]
    x_exp_bits = config["data_in_exponent_width"]
    x_frac_bits = config["data_in_frac_width"]
    clip_search = config.get("clip_search", False)
    force_fp32_had = config.get("force_fp32_had", False)

    def x_quantizer(t: Tensor) -> Tensor:
        return mxfp_rotate_quantizer(
            t,
            hadamard_dim=t.shape[-1],
            block_size=x_block_size,
            element_exp_bits=x_exp_bits,
            element_frac_bits=x_frac_bits,
            block_dim=-1,
            quantile_search=clip_search,
            force_fp32=force_fp32_had,
        )

    return x_quantizer(key_states), x_quantizer(value_states)
