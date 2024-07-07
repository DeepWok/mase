import torch

from chop.nn.quantized.modules import quantized_module_map
from ...module_modify_helper import replace_by_name, instantiate_module


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def quantize_by_type(network, pass_args):
    for type_name, config in pass_args.items():
        n_m = {}
        for n, m in network.named_modules():
            n_m[n] = m

        if type_name == "linear":
            module = torch.nn.Linear
        elif type_name == "conv2d":
            module = torch.nn.Conv2d
        else:
            raise ValueError(f"{type_name} is not supported!")
        config = config["config"]
        postfix = config.pop("name")
        for n, m in n_m.items():
            if isinstance(m, module):
                new_m = instantiate_module(
                    m, postfix, quantized_module_map, {"config": config}
                )
                network = replace_by_name(network, n, new_m)
    return network


def quantize_by_name(network, pass_args):
    quantize_names = pass_args.keys()
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m
    for n, m in n_m.items():
        if n in quantize_names:
            quan_config = pass_args[n]

            quan_config = quan_config["config"]
            postfix = quan_config.pop("name")

            new_m = instantiate_module(
                m, postfix, quantized_module_map, {"config": quan_config}
            )
            network = replace_by_name(network, n, new_m)
    return network


def quantize_module_transform_pass(network, pass_args):
    """
    Apply quantization transformation to the given nn.Module.

    :param network: The input network to be transformed.
    :type network: torch.nn.Module

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    Examples pass_args:

    .. code-block:: python

        pass_args = {
            "by": "type", # quantize by type, name, or regex_name
            "default": {"config": {"name": None}}, # default config, this would be used for any node that does not have a specific config
            "linear": {
                "config": {
                    "name": "integer",  # quantization scheme name supported are ["integer", "fixed" (equivalent to integer), "lutnet" (dev mode), "logicnets" (dev mode), "binary", "binary_residual", "ternary", "minifloat_ieee", "minifloat_denorm", "log", "block_fp", "block_minifloat", "block_log"]
                    # data
                    "data_in_width": 8,
                    "data_in_frac_width": 4,
                    # weight
                    "weight_width": 8,
                    "weight_frac_width": 4,
                    # bias
                    "bias_width": 8,
                    "bias_frac_width": 4,
                }
            },
        }

    :return: The transformed torch.nn.Module.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.

    """
    by = pass_args.pop("by")
    match by:
        case "type":
            network = quantize_by_type(network, pass_args)
        case "name":
            network = quantize_by_name(network, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')
    return network, {}
