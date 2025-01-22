import torch

from functools import reduce, partial
from copy import deepcopy
import logging

from transformers.models.roberta.modeling_roberta import (
    RobertaSdpaSelfAttention,
    RobertaClassificationHead,
    RobertaIntermediate,
    RobertaOutput,
    RobertaSelfOutput,
)

roberta_prefix_map = {
    RobertaSdpaSelfAttention: "roberta_self_attention",
    RobertaIntermediate: "roberta_intermediate",
    RobertaOutput: "roberta_output",
    RobertaSelfOutput: "roberta_self_output",
    RobertaClassificationHead: "roberta_classification_head",
}


def weight_replacement(x, y):
    target_state_dict = deepcopy(x.state_dict())
    missing_keys, unexpected_keys = y.load_state_dict(target_state_dict, strict=False)
    if missing_keys:
        logging.warning(
            f"Missing keys when loading state_dict: {missing_keys} from {x} to {y}"
        )
    if unexpected_keys:
        logging.warning(
            f"Unexpected keys when loading state_dict: {unexpected_keys} from {x} to {y}"
        )
    return y


def get_module_by_name(network, name):
    return network.get_submodule(name)
    # names = name.split(sep='.')
    # return reduce(getattr, names, module)


def set_module_by_name(
    model, name, target_module, parent_name=None, current_name=None, parent_model=None
):
    if name == parent_name:
        setattr(parent_model, current_name, target_module)
        return model

    for n, module in model.named_children():
        ## compound module, go inside it
        new_parent_name = n if parent_name is None else f"{parent_name}.{n}"
        set_module_by_name(module, name, target_module, new_parent_name, n, model)
    return model


def replace_by_name(network, name, module):
    original = get_module_by_name(network, name)
    new = weight_replacement(original, module)
    network = set_module_by_name(network, name, new)
    return network


"""
instantiation of different supported modules
"""


def instantiate_linear(module, postfix, module_map, additional_module_args):
    # if isinstance(module, torch.nn.Linear):
    linear_cls = module_map[f"linear_{postfix}"]
    has_bias = not (module.bias is None)
    linear = linear_cls(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=has_bias,
        **additional_module_args,
    )
    return linear


def instantiate_conv2d(module, postfix, module_map, additional_module_args):
    conv2d_cls = module_map[f"conv2d_{postfix}"]
    has_bias = not (module.bias is None)
    conv2d = conv2d_cls(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=has_bias,
        padding_mode=module.padding_mode,
        **additional_module_args,
    )
    return conv2d


def instantiate_roberta_module(module, postfix, module_map, additional_module_args):
    prefix = roberta_prefix_map[type(module)]
    roberta_cls = module_map[f"{prefix}_{postfix}"]

    model_config = additional_module_args["network_config"]
    q_config = additional_module_args["config"]

    roberta_module = roberta_cls(
        config=model_config,
        q_config=q_config,
    )
    return roberta_module


def instantiate_module(module, postfix, module_map, additional_module_args):
    if isinstance(module, torch.nn.Linear):
        module = instantiate_linear(module, postfix, module_map, additional_module_args)
    elif isinstance(module, torch.nn.Conv2d):
        module = instantiate_conv2d(module, postfix, module_map, additional_module_args)
    elif type(module) in roberta_prefix_map.keys():
        module = instantiate_roberta_module(
            module, postfix, module_map, additional_module_args
        )
    else:
        raise ValueError(f"{module} is not supported.")
    return module
