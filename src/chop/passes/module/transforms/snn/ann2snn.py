from chop.nn.quantizers.SNN.LSQ import LSQInteger
import torch

from chop.nn.snn.modules import spiking_module_map
from ...module_modify_helper import (
    manual_instantiate_module,
    replace_by_name,
    instantiate_module,
)
from ...state_dict_map import match_a_pattern, check_is_huggingface_model


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def convert_by_type(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)

    for type_name, conversion_config in pass_args.items():
        n_m = {}
        for n, m in network.named_modules():
            n_m[n] = m

        if type_name == "linear":
            module = torch.nn.Linear
        elif type_name == "conv2d":
            module = torch.nn.Conv2d
        elif type_name == "embedding":
            module = torch.nn.Embedding
        elif type_name == "layernorm":
            module = torch.nn.LayerNorm
        elif type_name == "relu":
            module = torch.nn.ReLU
        elif type_name == "lsqinteger":
            module = LSQInteger
        else:
            raise ValueError(f"{type_name} is not supported!")

        is_manual_instantiate = conversion_config.get("manual_instantiate", False)
        conversion_config = conversion_config["config"]
        postfix = conversion_config.pop("name")

        for n, m in n_m.items():
            if isinstance(m, module):
                # same across all convert methods
                additional_module_args = (
                    {"config": conversion_config, "network_config": network.config}
                    if is_huggingface_model
                    else {"config": conversion_config}
                )

                if is_manual_instantiate:
                    new_m = manual_instantiate_module(
                        m, postfix, spiking_module_map, additional_module_args
                    )
                else:
                    new_m = instantiate_module(
                        m, postfix, spiking_module_map, additional_module_args
                    )
                network = replace_by_name(network, n, new_m)

    return network


def convert_by_name(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)
    is_manual_instantiate = pass_args.get("manual_instantiate", False)

    conversion_names = pass_args.keys()
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m

    for n, m in n_m.items():
        if n in conversion_names:
            conversion_config = pass_args[n]

            conversion_config = conversion_config["config"]
            postfix = conversion_config.pop("name")

            # same across all convert methods
            additional_module_args = (
                {"config": conversion_config, "network_config": network.config}
                if is_huggingface_model
                else {"config": conversion_config}
            )

            if is_manual_instantiate:
                new_m = manual_instantiate_module(
                    m, postfix, spiking_module_map, additional_module_args
                )
            else:
                new_m = instantiate_module(
                    m, postfix, spiking_module_map, additional_module_args
                )
            network = replace_by_name(network, n, new_m)

    return network


def convert_by_regex_name(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)
    is_manual_instantiate = pass_args.get("manual_instantiate", False)

    patterns = list(pass_args.keys())
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m

    for n, m in n_m.items():
        matched_pattern = match_a_pattern(n, patterns)
        if not matched_pattern:
            continue

        conversion_config = pass_args[matched_pattern]["config"]
        postfix = conversion_config["name"]

        # same across all convert methods
        additional_module_args = (
            {"config": conversion_config, "network_config": network.config}
            if is_huggingface_model
            else {"config": conversion_config}
        )

        if is_manual_instantiate:
            new_m = manual_instantiate_module(
                m, postfix, spiking_module_map, additional_module_args
            )
        else:
            new_m = instantiate_module(
                m, postfix, spiking_module_map, additional_module_args
            )
        network = replace_by_name(network, n, new_m)

    return network


def ann2snn_module_transform_pass(network, pass_args):
    """
    Apply spike neural network (SNN) transformation to the input network.

    :param network: The input network to be transformed.
    :type network: torch.nn.Module

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    Examples pass_args:

    .. code-block:: python

        pass_args = {
            "by": "type", # transform by type, name, or regex_name
            "default": {"config": {"name": None}},
            "linear": {
                "config": {
                    "name": "unfold_bias",
                }
            },
        }

    :return: The transformed torch.nn.Module.
    :rtype: tuple
    :raises ValueError: If the convert "by" argument is unsupported.

    """
    by = pass_args.pop("by")
    match by:
        case "type":
            network = convert_by_type(network, pass_args)
        case "name":
            network = convert_by_name(network, pass_args)
        case "regex_name":
            network = convert_by_regex_name(network, pass_args)
        case _:
            raise ValueError(f'Unsupported conversion "by": {by}')
    return network, {}
