import os
import re
from copy import deepcopy
from dataclasses import dataclass

import toml
from chop.tools.config_load import convert_str_na_to_none

from ..quant_utils import parse_node_config

"""
An example of quant_config for llama

{
    "model_layer": {
        "self_attn": {
            "q_proj": {},
            "k_proj": {},
            "v_proj": {},
            "o_proj": {},
            "rotary_positional_encoding": {},
            "matmul_0": {},
            "matmul_1": {},
        },
        "mlp": {
            "gate_proj": {},
            "down_proj": {},
            "up_proj": {},
        },
    }
    "linear_default": {},
    "matmul_default": {},

}
"""


def create_layer_config(
    linear_qc: dict = None,
    matmul_qc: dict = None,
    rotary_positional_encoding_qc: dict = None,
    layer_qc: dict = None,
) -> dict:
    """
    Given the quantization config for linear, matmul, and RoPE, create a LlamaQuantizedDecoderLayer quantization config

    Automatically fill in the missing config with the default config
    """
    if (layer_qc is None and matmul_qc is None) and layer_qc is None:
        raise ValueError("Must provide either (linear_qc & matmul_qc) or layer_qc")
    if layer_qc is None:
        layer_qc = {}
    # fmt: off
    qc = {
        "self_attn": {
            "q_proj": parse_node_config(layer_qc.get("self_attn", {}).get("q_proj", linear_qc), "linear"),
            "k_proj": parse_node_config(layer_qc.get("self_attn", {}).get("k_proj", linear_qc), "linear"),
            "v_proj": parse_node_config(layer_qc.get("self_attn", {}).get("v_proj", linear_qc), "linear"),
            "o_proj": parse_node_config(layer_qc.get("self_attn", {}).get("o_proj", linear_qc), "linear"),
            "rotary_positional_encoding": parse_node_config(layer_qc.get("self_attn", {}).get("rotary_positional_encoding", rotary_positional_encoding_qc), "rotary_positional_encoding"),
            "matmul_0": parse_node_config(layer_qc.get("self_attn", {}).get("matmul_0", matmul_qc), "matmul"),
            "matmul_1": parse_node_config(layer_qc.get("self_attn", {}).get("matmul_1", matmul_qc), "matmul"),
        },
        "mlp": {
            "gate_proj": parse_node_config(layer_qc.get("mlp", {}).get("gate_proj", linear_qc), "linear"),
            "down_proj": parse_node_config(layer_qc.get("mlp", {}).get("down_proj", linear_qc), "linear"),
            "up_proj": parse_node_config(layer_qc.get("mlp", {}).get("up_proj", linear_qc), "linear")
        },
    }
    # fmt: on
    return qc


def by_type_parser(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config for by_class_parser"
    default_qc: dict = config["default"]
    linear_qc: dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    rotary_positional_encoding_qc: dict = parse_node_config(
        config.get("rotary_positional_encoding", default_qc),
        mase_op="rotary_positional_encoding",
    )
    matmul_qc: dict = parse_node_config(
        config.get("matmul", default_qc), mase_op="matmul"
    )
    layer_qc: dict = config.get("model_layer", None)

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = create_layer_config(
            linear_qc, matmul_qc, rotary_positional_encoding_qc, layer_qc
        )
    p_config["default"] = default_qc
    return p_config


def by_name_parser(config: dict, num_hidden_layers: int) -> dict:
    assert "default" in config, "Must provide default config for by_name_parser"
    default_qc: dict = config["default"]
    linear_qc: dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    rotary_positional_encoding_qc: dict = parse_node_config(
        config.get("rotary_positional_encoding", default_qc),
        mase_op="rotary_positional_encoding",
    )
    matmul_qc: dict = parse_node_config(
        config.get("matmul", default_qc), mase_op="matmul"
    )

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_qc = config.get(layer_entry, None)
        p_config[layer_entry] = create_layer_config(
            linear_qc, matmul_qc, rotary_positional_encoding_qc, layer_qc
        )
    p_config["default"] = default_qc
    return p_config


def parse_llama_quantized_config(config: str | dict, num_hidden_layers: int) -> dict:
    assert isinstance(
        config, (str, dict)
    ), "config must be a str path to config toml or dict"
    if isinstance(config, str):
        config = toml.load(config)
    config = convert_str_na_to_none(config)
    by = config.get("by", "type")
    match by:
        case "type":
            return by_type_parser(config, num_hidden_layers)
        case "name":
            return by_name_parser(config, num_hidden_layers)
        case _:
            raise ValueError(f"Unknown by: {by}")
