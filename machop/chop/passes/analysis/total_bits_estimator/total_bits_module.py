import logging
import numpy as np
from torch import nn

logger = logging.getLogger(__name__)


def total_bits_module_analysis_pass(module, pass_args: dict = {}):
    """
    Profile statistics analysis pass, a simplified, toy pass for now
    """
    assert isinstance(module, nn.Module), "module must be a nn.Module instance"
    assert isinstance(pass_args, dict), "pass_args must be a dict instance"

    weights_size, weight_bits = 0, 0

    for n, m in module.named_modules():
        # a simple estimation to loop around only linear layers
        if isinstance(m, nn.Linear) and hasattr(m, "config"):
            weights_size += m.in_features * m.out_features
            weight_bits += m.in_features * m.out_features * m.config["weight_width"]

    if weight_bits == 0:
        logger.warning(
            "No quantized layers found in the model, set average_bitwidth to 32"
        )
        pass_args |= {"average_bitwidth": 32}
        return module
    else:
        pass_args |= {"average_bitwidth": weight_bits / weights_size}
        return module
