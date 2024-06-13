from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from .utils import get_stats, quantiser_passthrough

from chop.nn.quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
)


class _SiLUBase(torch.nn.SiLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.silu(x)
        else:
            x = self.x_quantizer(x)
            return F.silu(x, self.inplace)

    def get_quantized_output(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        return {"x": x}


class SiLUInteger(_SiLUBase):
    bypass = None

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizers
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width, is_signed=False
        )
        self.config = config
        self.x_width = x_width
        self.x_frac_width = x_frac_width


class SiLUMinifloatDenorm(_SiLUBase):
    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        self.x_quantizer = partial(
            minifloat_denorm_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )


class SiLUMinifloatIEEE(_SiLUBase):
    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        self.x_quantizer = partial(
            minifloat_ieee_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )


class SiLULog(_SiLUBase):
    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        x_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_bias"],
        )
        self.x_quantizer = partial(
            log_quantizer,
            width=x_width,
            exponent_bias=x_exponent_bias,
        )


class SiLULog(_SiLUBase):
    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        x_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_bias"],
        )
        self.x_quantizer = partial(
            log_quantizer,
            width=x_width,
            exponent_bias=x_exponent_bias,
        )


class SiLUBlockFP(_SiLUBase):
    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        x_width, x_exponent_width, x_exponent_bias, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
            config["data_in_block_size"],
        )
        self.x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.silu(x)
        else:
            x_shape = [i for i in x.shape]
            if x.ndim > 2:
                x = torch.flatten(x, 0, -3)
            x = self.x_quantizer(x)
            x = torch.reshape(x, x_shape)
            return F.silu(x, self.inplace)


class SiLUBlockMinifloat(_SiLUBase):
    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        x_width, x_exponent_width, x_exponent_bias_width, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )
        self.x_quantizer = partial(
            block_minifloat_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.silu(x)
        else:
            x_shape = [i for i in x.shape]
            if x.ndim > 2:
                x = torch.flatten(x, 0, -3)
            x = self.x_quantizer(x)
            x = torch.reshape(x, x_shape)
            return F.silu(x, self.inplace)


class SiLUBlockLog(_SiLUBase):
    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)

        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        x_width, x_exponent_bias_width, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )
        self.x_quantizer = partial(
            block_log_quantizer,
            width=x_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.silu(x)
        else:
            x_shape = [i for i in x.shape]
            if x.ndim > 2:
                x = torch.flatten(x, 0, -3)
            x = self.x_quantizer(x)
            x = torch.reshape(x, x_shape)
            return F.silu(x, self.inplace)


class SiLUBinary(_SiLUBase):
    bypass = None

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizers
        x_stochastic = config["data_in_stochastic"]
        x_bipolar = config["data_in_bipolar"]
        self.x_quantizer = quantiser_passthrough
        # self.x_quantizer = partial(
        #     binary_quantizer, stochastic=x_stochastic, bipolar=x_bipolar
        # )
        self.config = config
        # self.x_width = x_width
        # self.x_frac_width = x_frac_width


class SiLUTernary(_SiLUBase):
    bypass = None

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantisers
        x_scaling_factor = config["data_in_scaling_factor"]
        x_mean = get_stats(config, "data_in_mean")
        x_median = get_stats(config, "data_in_median")
        x_max = get_stats(config, "data_in_max")
        self.x_quantizer = quantiser_passthrough
        # self.x_quantizer = partial(
        #     ternary_quantizer,
        #     scaling_factor=x_scaling_factor,
        #     median=x_median,
        #     maximum=x_max,
        #     mean=x_mean,
        # )
        self.config = config
