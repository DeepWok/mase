import torch
from transformers.models.bert.modeling_bert import(
    BertSelfAttention, 
    BertSdpaSelfAttention, 
    BertSelfOutput, 
    BertAttention
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2SdpaAttention,
    GPT2Block,
)
from chop.nn.attention.modules import attention_module_map
from ...module_modify_helper import replace_by_name, instantiate_module
from ...state_dict_map import match_a_pattern, check_is_huggingface_model
from .attention_transform_helper import MLAWrapper, llama2_to_mla_init, transform_llama2_to_mla

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer
)

def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def mla_by_type(network, pass_args):
    # Import the necessary classes
    from transformers.models.llama.modeling_llama import LlamaAttention
    from .attention_transform_helper import MLAAttentionWrapper, transform_llama2_to_mla, llama2_to_mla_init
    
    transformed_count = 0
    stats = {}  # Create stats dictionary to return
    
    for type_name, config_entry in pass_args.items():
        # Collect all modules
        n_m = {}
        for n, m in network.named_modules():
            n_m[n] = m
            
        # Extract config - handle both nested and flat config formats
        if "config" in config_entry:
            config = config_entry["config"].copy()
        else:
            config = config_entry.copy() 
            
        # Get postfix, defaulting to "mla" if not specified
        postfix = config.pop("name", "mla")
        print(f"Using postfix: {postfix}, config: {config}")
        
        stats[type_name] = {"transformed_modules": []}  # Track transformed modules in stats
            
        if type_name == "llama":
            print(f"Looking for Llama attention modules...")
            
            # Find and transform all matching modules
            for n, m in n_m.items():
                # Check if it's an attention module (by any detection method)
                is_attention = False
                if isinstance(m, LlamaAttention):
                    is_attention = True
                    print(f"Found exact match: {n}")
                elif "Attention" in type(m).__name__ and "Llama" in type(m).__name__:
                    is_attention = True
                    print(f"Found name match: {n}")
                    
                if is_attention:
                    try:
                        # Create MLA module (using helper functions directly)
                        mla_module = llama2_to_mla_init(m, {"config": config})
                        
                        # Transform weights
                        mla_module = transform_llama2_to_mla(m, mla_module)
                        
                        # Create wrapper with proper interface 
                        wrapped_module = MLAAttentionWrapper(mla_module)
                        
                        # Replace in model - find parent module and set attribute
                        if '.' in n:
                            parent_name, child_name = n.rsplit('.', 1)
                            parent = network
                            for part in parent_name.split('.'):
                                parent = getattr(parent, part)
                            setattr(parent, child_name, wrapped_module)
                        else:
                            setattr(network, n, wrapped_module)
                            
                        transformed_count += 1
                        stats[type_name]["transformed_modules"].append(n)
                        print(f"Successfully transformed {n}")
                    except Exception as e:
                        print(f"Error transforming {n}: {str(e)}")
                        import traceback
                        traceback.print_exc()
            
            # Also look for decoder layers that have self_attn
            for n, m in n_m.items():
                if "DecoderLayer" in type(m).__name__ and hasattr(m, "self_attn"):
                    attn_module = m.self_attn
                    if not hasattr(attn_module, 'is_mla_wrapper'):  # Skip if already transformed
                        try:
                            # Create MLA module
                            mla_module = llama2_to_mla_init(attn_module, {"config": config})
                            
                            # Transform weights
                            mla_module = transform_llama2_to_mla(attn_module, mla_module)
                            
                            # Create wrapper with proper interface
                            wrapped_module = MLAAttentionWrapper(mla_module)
                            
                            # Replace directly in the decoder layer
                            m.self_attn = wrapped_module
                            
                            transformed_count += 1
                            stats[type_name]["transformed_modules"].append(f"{n}.self_attn")
                            print(f"Successfully transformed {n}.self_attn")
                        except Exception as e:
                            print(f"Error transforming {n}.self_attn: {str(e)}")
                            import traceback
                            traceback.print_exc()
            
            # Skip the rest of the loop for "llama" type
            continue
                            
        # Rest of function for other module types...
        # ...
        
    print(f"Transformed {transformed_count} modules in total")
    stats["total_transformed"] = transformed_count
    return network, stats  # Return both network and stats

def mla_by_name(network, pass_args):
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


def mla_by_regex_name(network, pass_args):
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


def attention_transform_pass(network, pass_args):
    by = pass_args.pop("by")
    stats = {}
    match by:
        case "type":
            network, type_stats = mla_by_type(network, pass_args)
            stats.update(type_stats)
        case "name":
            network = mla_by_name(network, pass_args)
        case "regex_name":
            network = mla_by_regex_name(network, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')
    return network, stats


