import torch
import transformers
from chop.nn.replaced.modules import replaced_module_map
from ...state_dict_map import match_a_pattern, check_is_huggingface_model
from clip.model import QuickGELU

from ...module_modify_helper import (
    manual_instantiate_module,
    replace_by_name,
    instantiate_module,
)

def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def replaced_by_type(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)

    for type_name, replace_config in pass_args.items():
        n_m = {}
        for n, m in network.named_modules():
            n_m[n] = m

        if type_name == "gelu":
            module = QuickGELU
        else:
            raise ValueError(f"{type_name} is not supported!")
        is_manual_instantiate = replace_config.get("manual_instantiate", False)
        replace_config = replace_config["config"]
        postfix = replace_config.pop("name")

        for n, m in n_m.items():
            if isinstance(m, module):
                # same across all convert methods
                additional_module_args = (
                    {"config": replace_config, "network_config": network.config}
                    if is_huggingface_model
                    else {"config": replace_config}
                )

                if is_manual_instantiate:
                    new_m = manual_instantiate_module(
                        m, postfix, replaced_module_map, additional_module_args
                    )
                else:
                    new_m = instantiate_module(
                        m, postfix, replaced_module_map, additional_module_args
                    )
                network = replace_by_name(network, n, new_m)

    return network


def replaced_by_name(network, pass_args):
    pass
    # is_huggingface_model = check_is_huggingface_model(network)

    # replace_names = pass_args.keys()
    # n_m = {}
    # for n, m in network.named_modules():
    #     n_m[n] = m
    # for n, m in n_m.items():
    #     if n in replace_names:
    #         quan_config = pass_args[n]

    #         quan_config = quan_config["config"]
    #         postfix = quan_config.pop("name")

    #         additional_module_args = (
    #             {"config": quan_config, "network_config": network.config}
    #             if is_huggingface_model
    #             else {"config": quan_config}
    #         )

    #         new_m = instantiate_module(
    #             m, postfix, replaced_module_map, additional_module_args
    #         )
    #         network = replace_by_name(network, n, new_m)
    # return network


def replaced_by_regex_name(network, pass_args):
    pass
    # is_huggingface_model = check_is_huggingface_model(network)

    # patterns = list(pass_args.keys())
    # n_m = {}
    # for n, m in network.named_modules():
    #     n_m[n] = m

    # for n, m in n_m.items():
    #     matched_pattern = match_a_pattern(n, patterns)
    #     if not matched_pattern:
    #         continue

    #     quan_config = pass_args[matched_pattern]["config"]
    #     postfix = quan_config["name"]

    #     additional_module_args = (
    #         {"config": quan_config, "network_config": network.config}
    #         if is_huggingface_model
    #         else {"config": quan_config}
    #     )

    #     new_m = instantiate_module(
    #         m, postfix, replaced_module_map, additional_module_args
    #     )
    #     network = replace_by_name(network, n, new_m)

    # return network


def replace_module_transform_pass(network, pass_args):
    """
    Apply quantization transformation to the given nn.Module.

    :param network: The input network to be transformed.
    :type network: torch.nn.Module

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    Examples pass_args:

    .. code-block:: python

        pass_args = {
            "by": "type", # replace by type, name, or regex_name
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
    :raises ValueError: If the replace "by" argument is unsupported.

    """
    by = pass_args.pop("by")
    match by:
        case "type":
            network = replaced_by_type(network, pass_args)
        case "name":
            network = replaced_by_name(network, pass_args)
        case "regex_name":
            network = replaced_by_regex_name(network, pass_args)
        case _:
            raise ValueError(f'Unsupported replace "by": {by}')
    return network, {}
