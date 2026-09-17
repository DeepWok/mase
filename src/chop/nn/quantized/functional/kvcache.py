from functools import partial

from torch import Tensor

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer


def _mx_tensor(
    tensor: Tensor,
    config: dict,
    *,
    rotate: bool = False,
) -> Tensor:
    """Quantize one cache operand from its role-specific MX config."""

    cfg = config or {}
    has_int = "data_in_width" in cfg
    has_fp = (
        "data_in_exponent_width" in cfg and "data_in_frac_width" in cfg
    )
    if has_int == has_fp:
        raise ValueError(
            "KV precision must define exactly one MXINT or MXFP element format"
        )
    block_size = cfg["data_in_block_size"]
    if has_int:
        if rotate:
            from chop.nn.quantizers.rotation import mxint_rotate_quantizer

            return mxint_rotate_quantizer(
                tensor,
                hadamard_dim=tensor.shape[-1],
                block_size=block_size,
                element_bits=cfg["data_in_width"],
                block_dim=-1,
                quantile_search=cfg.get("clip_search", False),
                force_fp32=cfg.get("force_fp32_had", False),
            )
        return mxint_quantizer(
            tensor,
            block_size=block_size,
            element_bits=cfg["data_in_width"],
            block_dim=-1,
        )
    if rotate:
        from chop.nn.quantizers.rotation import mxfp_rotate_quantizer

        return mxfp_rotate_quantizer(
            tensor,
            hadamard_dim=tensor.shape[-1],
            block_size=block_size,
            element_exp_bits=cfg["data_in_exponent_width"],
            element_frac_bits=cfg["data_in_frac_width"],
            block_dim=-1,
            quantile_search=cfg.get("clip_search", False),
            force_fp32=cfg.get("force_fp32_had", False),
        )
    return mxfp_quantizer(
        tensor,
        block_size=block_size,
        element_exp_bits=cfg["data_in_exponent_width"],
        element_frac_bits=cfg["data_in_frac_width"],
        block_dim=-1,
    )


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


def kv_cache_mx(
    key_states: Tensor,
    value_states: Tensor,
    config: dict = None,
    *,
    rotate: bool = False,
) -> tuple[Tensor, Tensor]:
    """Dispatch KV storage independently from the attention A-side format."""

    cfg = config or {}
    if "key" in cfg or "value" in cfg:
        if set(cfg) != {"key", "value"}:
            raise ValueError(
                "split KV precision requires exactly key and value configs"
            )
        return kv_cache_mx_split(
            key_states,
            value_states,
            key_config=cfg["key"],
            value_config=cfg["value"],
            rotate=rotate,
        )
    has_int = "data_in_width" in cfg
    has_fp = (
        "data_in_exponent_width" in cfg and "data_in_frac_width" in cfg
    )
    if has_int == has_fp:
        raise ValueError(
            "KV precision must define exactly one MXINT or MXFP element format"
        )
    if has_int:
        quantizer = kv_cache_mxint_rotate if rotate else kv_cache_mxint
    else:
        quantizer = kv_cache_mxfp_rotate if rotate else kv_cache_mxfp
    return quantizer(key_states, value_states, cfg)


def kv_cache_mx_split(
    key_states: Tensor,
    value_states: Tensor,
    *,
    key_config: dict,
    value_config: dict,
    rotate: bool = False,
) -> tuple[Tensor, Tensor]:
    """Quantize K and V independently while preserving their tensor roles."""

    return (
        _mx_tensor(key_states, key_config, rotate=rotate),
        _mx_tensor(value_states, value_config, rotate=rotate),
    )


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
