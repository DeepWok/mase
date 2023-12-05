import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def total_bits_module_analysis_pass(
    module: torch.nn.Module, pass_args: dict = {}
) -> tuple:
    """
    Analyzes the total number of bits required by the quantized layers in a given module.

    :param module: The module to analyze.
    :type module: torch.nn.Module
    :param pass_args: Additional arguments for the analysis pass. (default: {})
    :type pass_args: dict

    :return: A tuple containing the modified module and a dictionary with the analysis results.
    :rtype: tuple(MaseGraph, dict)
    """

    assert isinstance(module, torch.nn.Module), "module must be a nn.Module instance"
    assert isinstance(pass_args, dict), "pass_args must be a dict instance"
    return_info = {}

    weights_size, weight_bits = 0, 0

    for n, m in module.named_modules():
        # a simple estimation to loop around only linear layers
        if isinstance(m, torch.nn.Linear) and hasattr(m, "config"):
            weights_size += m.in_features * m.out_features
            weight_bits += m.in_features * m.out_features * m.config["weight_width"]

    if weight_bits == 0:
        logger.warning(
            "No quantized layers found in the model, set average_bitwidth to 32"
        )
        return_info |= {"average_bitwidth": 32}
        return module, return_info
    else:
        return_info |= {"average_bitwidth": weight_bits / weights_size}
        return module, return_info
