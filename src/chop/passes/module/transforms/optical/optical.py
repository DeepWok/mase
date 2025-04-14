import torch

from chop.nn.optical.modules import optical_module_map
from chop.passes.module.module_modify_helper import instantiate_module
from chop.passes.module.transforms.optical.module_transform_helper import (
    replace_by_name_optical,
)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def optical_transform_by_type(network, pass_args):
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
                    m, postfix, optical_module_map, {"config": config}
                )
                network = replace_by_name_optical(network, n, new_m)
    return network


def optical_transform_by_name(network, pass_args):
    optical_names = pass_args.keys()
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m
    for n, m in n_m.items():
        if n in optical_names:
            optical_config = pass_args[n]

            optical_config = optical_config["config"]
            postfix = optical_config.pop("name")

            new_m = instantiate_module(
                m, postfix, optical_module_map, {"config": optical_config}
            )
            network = replace_by_name_optical(network, n, new_m)
    return network


def optical_module_transform_pass(network, pass_args):
    """
    Apply optical transformation to the given nn.Module.

    :param network: The input network to be transformed.
    :type network: torch.nn.Module

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    :return: The transformed torch.nn.Module.
    :rtype: tuple
    :raises ValueError: If the "by" argument is unsupported.
    """
    by = pass_args.pop("by")
    match by:
        case "type":
            network = optical_transform_by_type(network, pass_args)
        case "name":
            network = optical_transform_by_name(network, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')
    return network, {}
