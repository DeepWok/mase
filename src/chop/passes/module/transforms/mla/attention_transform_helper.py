from ...module_modify_helper import get_module_by_name, set_module_by_name

# TODO: Merge into module modify helpers

def instantiate_attention_module(module, postfix, module_map, additional_module_args):

    module_args = additional_module_args["config"]
    network_args = additional_module_args.get("network_config", None)

    module = instantiate_attention(module, postfix, module_map, module_args)

    return module


def instantiate_attention(module, postfix, module_map, additional_module_args):
    attention_cls = module_map[f"attention_{postfix}"]
    has_bias = not (module.bias is None)

    attention = attention_cls(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=has_bias,
        config=additional_module_args,
    )

    return attention

def replace_attention_by_name(network, name, module):
    original = get_module_by_name(network, name)

    # state_dict replacement
    special_replacement = (type(original), type(module)) in SPECIAL_CONVERT_PATTERNS
    if special_replacement:
        new = SPECIAL_CONVERT_PATTERNS[(type(original), type(module))](original, module)
    else:
        new = weight_replacement(original, module)

    network = set_module_by_name(network, name, new)
    return network