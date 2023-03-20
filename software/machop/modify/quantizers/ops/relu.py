from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from ....graph.mase_tracer import mark_as_leaf_module
from ..quantizers import integer_quantizer
from .utils import extract_required_config


@mark_as_leaf_module
class ReLUInteger(torch.nn.ReLU):
    _required_config_keys = ("name", "input_width", "input_frac_width")
    _optional_config_keys = ("bypass",)

    def __init__(self, inplace: bool = False, config: dict = {}):
        super().__init__(inplace)
        self.bypass = config.get("bypass", False)
        # establish quantizers
        x_width, x_frac_width = config["input_width"], config["input_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
        self.config = self.construct_essential_config(config)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.relu(x)
        x = self.x_quantizer(x)
        return F.relu(x)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config
