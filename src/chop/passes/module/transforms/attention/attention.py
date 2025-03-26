import torch
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertSdpaSelfAttention,
    BertSelfOutput,
    BertAttention,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2SdpaAttention,
    GPT2Block,
)
from chop.nn.attention.modules import attention_module_map
from ...module_modify_helper import replace_by_name, instantiate_module
from ...state_dict_map import match_a_pattern, check_is_huggingface_model
from .attention_transform_helper import (
    instantiate_attention_module,   
    replace_attention_by_name,
    transform_llama_to_mla,
 )
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer
)

def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def attention_by_type(network, pass_args):
    for type_name, config in pass_args.items():
        n_m = {}
        for n, m in network.named_modules():
            n_m[n] = m
        
        if type_name == "gpt2spda":
            module = GPT2SdpaAttention
        elif type_name == "gpt2block":
            module = GPT2Block
        else:
            raise ValueError(f"{type_name} is not supported!")
        config = config["config"]
        postfix = config.pop("name")
        transform_name = type_name + "_to_" + postfix
        for n, m in n_m.items():
            if isinstance(m, module):
                new_m = instantiate_attention_module(
                    m, transform_name, attention_module_map, {"config": config}
                )
                network = replace_attention_by_name(network, n, new_m, transform_name)
    return network

def attention_by_name(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)

    quantize_names = pass_args.keys()
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m
    for n, m in n_m.items():
        if n in quantize_names:
            quan_config = pass_args[n]

            quan_config = quan_config["config"]
            postfix = quan_config.pop("name")

            additional_module_args = (
                {"config": quan_config, "network_config": network.config}
                if is_huggingface_model
                else {"config": quan_config}
            )

            new_m = instantiate_module(
                m, postfix, attention_module_map, additional_module_args
            )
            network = replace_by_name(network, n, new_m)
    return network


def attention_by_regex_name(network, pass_args):
    is_huggingface_model = check_is_huggingface_model(network)

    patterns = list(pass_args.keys())
    n_m = {}
    for n, m in network.named_modules():
        n_m[n] = m

    for n, m in n_m.items():
        matched_pattern = match_a_pattern(n, patterns)
        if not matched_pattern:
            continue

        quan_config = pass_args[matched_pattern]["config"]
        postfix = quan_config["name"]

        additional_module_args = (
            {"config": quan_config, "network_config": network.config}
            if is_huggingface_model
            else {"config": quan_config}
        )

        new_m = instantiate_module(
            m, postfix, attention_module_map, additional_module_args
        )
        network = replace_by_name(network, n, new_m)

    return network

def attention_by_model(network, pass_args):
    for model_name, config in pass_args.items():
        if model_name == "llama":
            new_network = transform_llama_to_mla(network, config)
        else:
            raise ValueError(f"Model {model_name} is not supported!")
    return new_network

def attention_transform_pass(network, pass_args):
    by = pass_args.pop("by")
    stats = {}
    match by:
        case "type":
            network = attention_by_type(network, pass_args)
        case "name":
            network = attention_by_name(network, pass_args)
        case "regex_name":
            network = attention_by_regex_name(network, pass_args)
        case "model":
            network = attention_by_model(network, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')
    return network, stats


