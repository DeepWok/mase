import os
import re
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch import nn
import toml
from chop.tools.config_load import convert_str_na_to_none


def parse_node_config(config: dict, layer_type: str):
    match layer_type:
        case "linear":
            return {
                "r": config["r"],
                "lora_alpha": config["lora_alpha"],
                "lora_dropout": config["lora_dropout"],
            }
        case _:
            raise ValueError(
                f"Invalid layer_type: {layer_type}. Supported layer types are 'linear'."
            )


def create_a_layer_config(linear_lc: dict = None, layer_lc=None) -> dict:
    if linear_lc is None and layer_lc is None:
        raise ValueError("Must provide either linear_lc or layer_qc")
    if layer_lc is None:
        layer_lc = {}
    # fmt: off
    lc = {
        "self_attn": {
            "q_proj": parse_node_config(layer_lc.get("self_attn", {}).get("q_proj", linear_lc), "linear"),
            "k_proj": parse_node_config(layer_lc.get("self_attn", {}).get("k_proj", linear_lc), "linear"),
            "v_proj": parse_node_config(layer_lc.get("self_attn", {}).get("v_proj", linear_lc), "linear"),
            "o_proj": parse_node_config(layer_lc.get("self_attn", {}).get("o_proj", linear_lc), "linear"),
            
        },
    }
 
    return lc


def by_type_parser(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config for by_type_parser"
    default_lc: dict = config["default"]
    linear_lc: dict = parse_node_config(
        config.get("linear", default_lc), layer_type="linear"
    )

    layer_lc: dict = config.get("model_layer", None)

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = create_a_layer_config(linear_lc, layer_lc)
    p_config["default"] = default_lc
    return p_config


def by_name_parser(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config for by_name_parser"
    default_lc: dict = config["default"]
    linear_lc: dict = parse_node_config(
        config.get("linear", default_lc), layer_type="linear"
    )

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_lc = config.get(layer_entry, None)
        p_config[layer_entry] = create_a_layer_config(linear_lc, layer_lc)
    p_config["default"] = default_lc
    return p_config


def parse_llama_lora_config(config: str | dict, num_hidden_layers: int) -> dict:
    assert isinstance(
        config, (str, dict)
    ), "config must be a str path to config toml or dict"
    if isinstance(config, str):
        config = toml.load(config)
    config = convert_str_na_to_none(config)
    by = config.pop("by", "type")
    match by:
        case "type":
            return by_type_parser(config, num_hidden_layers)
        case "name":
            return by_name_parser(config, num_hidden_layers)
        case _:
            raise ValueError(f"Unknown by: {by}")
