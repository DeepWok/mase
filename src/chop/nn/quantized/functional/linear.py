from functools import partial
from torch import Tensor
from torch.nn import functional as F
from ..utils import get_stats, quantiser_passthrough

from chop.nn.quantizers import (
    residual_sign_quantizer,
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    integer_floor_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
    ternary_quantizer,
    mxint_hardware,
)


def linearInteger(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
    out_config: dict = None,
):
    # establish quantizer
    w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
    x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
    # check bias quantizer, if not, use weight quantizer
    b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]

    # TODO: remove out_config in the future (some tb are still using out_config), use config instead
    if out_config is not None:
        out_width, out_frac_width = (
            out_config["data_out_width"],
            out_config["data_out_frac_width"],
        )
    elif "data_out_width" in config:
        out_width, out_frac_width = (
            config["data_out_width"],
            config["data_out_frac_width"],
        )

    floor = config.get("floor", False)
    base_quantizer = integer_floor_quantizer if floor else integer_quantizer
    w_quantizer = partial(base_quantizer, width=w_width, frac_width=w_frac_width)
    x_quantizer = partial(base_quantizer, width=x_width, frac_width=x_frac_width)
    b_quantizer = partial(base_quantizer, width=b_width, frac_width=b_frac_width)

    if out_config is not None:
        out_quantizer = partial(
            base_quantizer, width=out_width, frac_width=out_frac_width
        )
    elif "data_out_width" in config:
        out_quantizer = partial(
            base_quantizer, width=out_width, frac_width=out_frac_width
        )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None
    out = F.linear(x, weight, bias)

    if "data_out_width" in config:
        out = out_quantizer(out)
    elif out_config is not None:
        out = out_quantizer(out)
    return out


def linearMinifloatDenorm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    w_width, w_exponent_width, w_exponent_bias = (
        config["weight_width"],
        config["weight_exponent_width"],
        config["weight_exponent_bias"],
    )
    x_width, x_exponent_width, x_exponent_bias = (
        config["data_in_width"],
        config["data_in_exponent_width"],
        config["data_in_exponent_bias"],
    )
    b_width, b_exponent_width, b_exponent_bias = (
        config["bias_width"],
        config["bias_exponent_width"],
        config["bias_exponent_bias"],
    )

    w_quantizer = partial(
        minifloat_denorm_quantizer,
        width=w_width,
        exponent_width=w_exponent_width,
        exponent_bias=w_exponent_bias,
    )

    x_quantizer = partial(
        minifloat_denorm_quantizer,
        width=x_width,
        exponent_width=x_exponent_width,
        exponent_bias=x_exponent_bias,
    )

    b_quantizer = partial(
        minifloat_denorm_quantizer,
        width=b_width,
        exponent_width=b_exponent_width,
        exponent_bias=b_exponent_bias,
    )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearMinifloatIEEE(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    w_width, w_exponent_width, w_exponent_bias = (
        config["weight_width"],
        config["weight_exponent_width"],
        config["weight_exponent_bias"],
    )
    x_width, x_exponent_width, x_exponent_bias = (
        config["data_in_width"],
        config["data_in_exponent_width"],
        config["data_in_exponent_bias"],
    )
    b_width, b_exponent_width, b_exponent_bias = (
        config["bias_width"],
        config["bias_exponent_width"],
        config["bias_exponent_bias"],
    )

    w_quantizer = partial(
        minifloat_ieee_quantizer,
        width=w_width,
        exponent_width=w_exponent_width,
        exponent_bias=w_exponent_bias,
    )

    x_quantizer = partial(
        minifloat_ieee_quantizer,
        width=x_width,
        exponent_width=x_exponent_width,
        exponent_bias=x_exponent_bias,
    )

    b_quantizer = partial(
        minifloat_ieee_quantizer,
        width=b_width,
        exponent_width=b_exponent_width,
        exponent_bias=b_exponent_bias,
    )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearMinifloatIEEE(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    w_width, w_exponent_width, w_exponent_bias = (
        config["weight_width"],
        config["weight_exponent_width"],
        config["weight_exponent_bias"],
    )
    x_width, x_exponent_width, x_exponent_bias = (
        config["data_in_width"],
        config["data_in_exponent_width"],
        config["data_in_exponent_bias"],
    )
    b_width, b_exponent_width, b_exponent_bias = (
        config["bias_width"],
        config["bias_exponent_width"],
        config["bias_exponent_bias"],
    )

    w_quantizer = partial(
        minifloat_ieee_quantizer,
        width=w_width,
        exponent_width=w_exponent_width,
        exponent_bias=w_exponent_bias,
    )

    x_quantizer = partial(
        minifloat_ieee_quantizer,
        width=x_width,
        exponent_width=x_exponent_width,
        exponent_bias=x_exponent_bias,
    )

    b_quantizer = partial(
        minifloat_ieee_quantizer,
        width=b_width,
        exponent_width=b_exponent_width,
        exponent_bias=b_exponent_bias,
    )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearLog(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    w_width, w_exponent_bias = (
        config["weight_width"],
        config["weight_exponent_bias"],
    )
    x_width, x_exponent_bias = (
        config["data_in_width"],
        config["data_in_exponent_bias"],
    )
    b_width, b_exponent_bias = (
        config["bias_width"],
        config["bias_exponent_bias"],
    )

    w_quantizer = partial(
        log_quantizer,
        width=w_width,
        exponent_bias=w_exponent_bias,
    )

    x_quantizer = partial(
        log_quantizer,
        width=x_width,
        exponent_bias=x_exponent_bias,
    )

    b_quantizer = partial(
        log_quantizer,
        width=b_width,
        exponent_bias=b_exponent_bias,
    )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearBlockFP(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    # establish quantizers
    w_width, w_exponent_width, w_exponent_bias, w_block_size = (
        config["weight_width"],
        config["weight_exponent_width"],
        config["weight_exponent_bias"],
        config["weight_block_size"],
    )
    x_width, x_exponent_width, x_exponent_bias, x_block_size = (
        config["data_in_width"],
        config["data_in_exponent_width"],
        config["data_in_exponent_bias"],
        config["data_in_block_size"],
    )
    x_skip_first_dim = config.get("data_in_skip_first_dim", True)

    b_width, b_exponent_width, b_exponent_bias, b_block_size = (
        config["bias_width"],
        config["bias_exponent_width"],
        config["bias_exponent_bias"],
        config["bias_block_size"],
    )

    # blocking/unblocking 4D kernel/feature map is not supported
    w_quantizer = partial(
        block_fp_quantizer,
        width=w_width,
        exponent_width=w_exponent_width,
        exponent_bias=w_exponent_bias,
        block_size=w_block_size,
        skip_first_dim=False,
    )
    x_quantizer = partial(
        block_fp_quantizer,
        width=x_width,
        exponent_width=x_exponent_width,
        exponent_bias=x_exponent_bias,
        block_size=x_block_size,
        skip_first_dim=x_skip_first_dim,
    )
    b_quantizer = partial(
        block_fp_quantizer,
        width=b_width,
        exponent_width=b_exponent_width,
        exponent_bias=b_exponent_bias,
        block_size=b_block_size,
        skip_first_dim=False,
    )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearBlockMinifloat(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    # establish quantizers
    w_width, w_exponent_width, w_exponent_bias_width, w_block_size = (
        config["weight_width"],
        config["weight_exponent_width"],
        config["weight_exponent_bias_width"],
        config["weight_block_size"],
    )
    x_width, x_exponent_width, x_exponent_bias_width, x_block_size = (
        config["data_in_width"],
        config["data_in_exponent_width"],
        config["data_in_exponent_bias_width"],
        config["data_in_block_size"],
    )
    x_skip_first_dim = config.get("data_in_skip_first_dim", True)

    b_width, b_exponent_width, b_exponent_bias_width, b_block_size = (
        config["bias_width"],
        config["bias_exponent_width"],
        config["bias_exponent_bias_width"],
        config["bias_block_size"],
    )

    # blocking/unblocking 4D kernel/feature map is not supported
    w_quantizer = partial(
        block_minifloat_quantizer,
        width=w_width,
        exponent_width=w_exponent_width,
        exponent_bias_width=w_exponent_bias_width,
        block_size=w_block_size,
        skip_first_dim=False,
    )
    x_quantizer = partial(
        block_minifloat_quantizer,
        width=x_width,
        exponent_width=x_exponent_width,
        exponent_bias_width=x_exponent_bias_width,
        block_size=x_block_size,
        skip_first_dim=x_skip_first_dim,
    )
    b_quantizer = partial(
        block_minifloat_quantizer,
        width=b_width,
        exponent_width=b_exponent_width,
        exponent_bias_width=b_exponent_bias_width,
        block_size=b_block_size,
        skip_first_dim=False,
    )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearBlockLog(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    # establish quantizers
    w_width, w_exponent_bias_width, w_block_size = (
        config["weight_width"],
        config["weight_exponent_bias_width"],
        config["weight_block_size"],
    )
    x_width, x_exponent_bias_width, x_block_size = (
        config["data_in_width"],
        config["data_in_exponent_bias_width"],
        config["data_in_block_size"],
    )
    x_skip_first_dim = config.get("data_in_skip_first_dim", True)

    b_width, b_exponent_bias_width, b_block_size = (
        config["bias_width"],
        config["bias_exponent_bias_width"],
        config["bias_block_size"],
    )

    # blocking/unblocking 4D kernel/feature map is not supported
    w_quantizer = partial(
        block_log_quantizer,
        width=w_width,
        exponent_bias_width=w_exponent_bias_width,
        block_size=w_block_size,
        skip_first_dim=False,
    )
    x_quantizer = partial(
        block_log_quantizer,
        width=x_width,
        exponent_bias_width=x_exponent_bias_width,
        block_size=x_block_size,
        skip_first_dim=x_skip_first_dim,
    )
    b_quantizer = partial(
        block_log_quantizer,
        width=b_width,
        exponent_bias_width=b_exponent_bias_width,
        block_size=b_block_size,
        skip_first_dim=False,
    )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearBinary(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    w_stochastic = config["weight_stochastic"]
    w_bipolar = config["weight_bipolar"]
    w_quantizer = partial(binary_quantizer, stochastic=w_stochastic, bipolar=w_bipolar)
    b_quantizer = quantiser_passthrough
    x_quantizer = quantiser_passthrough

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearBinaryScaling(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    """
    Binary scaling variant of the linear transformation layer.

        - "bypass": Bypass quantization for standard linear transformation.
        - "data_in_stochastic", "bias_stochastic", "weight_stochastic": Stochastic settings.
        - "data_in_bipolar", "bias_bipolar", "weight_bipolar": Bipolar settings.
        - "binary_training": Apply binary scaling during training.
    """
    x_stochastic, b_stochastic, w_stochastic = (
        config["data_in_stochastic"],
        config["bias_stochastic"],
        config["weight_stochastic"],
    )
    x_bipolar, b_bipolar, w_bipolar = (
        config["data_in_bipolar"],
        config["bias_bipolar"],
        config["weight_bipolar"],
    )

    binary_training = config["binary_training"]

    w_quantizer = partial(binary_quantizer, stochastic=w_stochastic, bipolar=w_bipolar)
    x_quantizer = partial(binary_quantizer, stochastic=x_stochastic, bipolar=x_bipolar)
    b_quantizer = partial(binary_quantizer, stochastic=b_stochastic, bipolar=b_bipolar)

    if binary_training:
        x = x_quantizer(x)
        w = w_quantizer(weight)
        bias = b_quantizer(bias) if bias is not None else None
        return F.linear(
            x,
            # w * self.gamma.abs(),
            w,
            bias,
        )
    else:
        weight.data.clamp_(-1, 1)
        return F.linear(
            x,
            # self.weight * self.gamma.abs(),
            weight,
            bias,
        )


def linearTernary(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    w_scaling_factor = config["weight_scaling_factor"]
    w_mean = get_stats(config, "weight_mean")
    w_median = get_stats(config, "weight_median")
    w_max = get_stats(config, "weight_max")
    w_quantizer = partial(
        ternary_quantizer,
        scaling_factor=w_scaling_factor,
        maximum=w_max,
        median=w_median,
        mean=w_mean,
    )
    x_quantizer = quantiser_passthrough
    b_quantizer = quantiser_passthrough

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    return F.linear(x, weight, bias)


def linearBinaryResidualSign(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    raise NotImplementedError


def linearLUT(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    raise NotImplementedError


def linearLogicNets(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
):
    raise NotImplementedError


def linearMXIntHardware(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    config: dict = None,
    out_config: dict = None,
):
    w_width, w_exponent_width = (
        config["weight_width"],
        config["weight_exponent_width"],
    )
    w_p1, w_p0 = (
        config["weight_parallelism"][0],
        config["weight_parallelism"][1],
    )
    x_width, x_exponent_width = (
        config["data_in_width"],
        config["data_in_exponent_width"],
    )
    x_p1, x_p0 = (
        config["data_in_parallelism"][0],
        config["data_in_parallelism"][1],
    )
    # check bias quantizer, if not, use weight quantizer
    b_width, b_exponent_width = config["bias_width"], config["bias_exponent_width"]
    b_p1, b_p0 = config["bias_parallelism"][0], config["bias_parallelism"][1]
    base_quantizer = mxint_hardware
    if out_config is not None:
        out_width, out_exponent_width = (
            config["data_out_width"],
            config["data_out_exponent_width"],
        )
        out_p1, out_p0 = (
            config["data_out_parallelism_dim_1"],
            config["data_out_parallelism_dim_0"],
        )
        out_quantizer = partial(
            base_quantizer,
            q_config={"width": out_width, "exponent_width": out_exponent_width},
            parallelism=[out_p1, out_p0],
        )
    w_quantizer = partial(
        base_quantizer,
        q_config={"width": w_width, "exponent_width": w_exponent_width},
        parallelism=[w_p1, w_p0],
    )
    x_quantizer = partial(
        base_quantizer,
        q_config={"width": x_width, "exponent_width": x_exponent_width},
        parallelism=[x_p1, x_p0],
    )
    b_quantizer = partial(
        base_quantizer,
        q_config={"width": b_width, "exponent_width": b_exponent_width},
        parallelism=[b_p1, b_p0],
    )

    x = x_quantizer(x)
    weight = w_quantizer(weight)
    bias = b_quantizer(bias) if bias is not None else None

    out = F.linear(x, weight, bias)
    if out_config is not None:
        out = out_quantizer(out)
    return out
