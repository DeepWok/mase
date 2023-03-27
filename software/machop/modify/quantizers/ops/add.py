from functools import partial
from typing import Dict

import torch

from ....graph.mase_tracer import mark_as_leaf_module
from ..quantizers import integer_quantizer
from .utils import extract_required_config


@mark_as_leaf_module
class AddInteger(torch.nn.Module):
    _required_config_keys = ("name", "data_in_width", "data_in_frac_width")
    _optional_config_keys = ("bypass",)

    def __init__(self, config):
        super().__init__()

        self.bypass = config.get("bypass", False)
        # establish quantizers
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
        self.config = self.construct_essential_config(config)

    def forward(self, x, y):
        x = self.x_quantizer(x)
        y = self.x_quantizer(y)
        return x + y

    def construct_essential_config(self, config: Dict):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config
