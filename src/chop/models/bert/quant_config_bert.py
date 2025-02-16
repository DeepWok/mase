import os
import re
from copy import deepcopy
import logging

import toml
from chop.tools.config_load import convert_str_na_to_none

from ..quant_utils import parse_node_config


logger = logging.getLogger(__name__)
"""
An example of quant_config for bert

{
    "default": {}
    "model_layer": {
        "attention": {
            "query": {},
            "key": {},
            "value": {},
            "output": {
                "dense": {},
            },
            "matmul_0": {},
            "matmul_1": {},
        },
        # TODO: does not support cross attention yet
        "crossattntion": { # if config.add_cross_attention is True
            "query": {},
            "key": {},
            "value": {},
            "output": {
                "dense": {},
            },
            "matmul_0": {},
            "matmul_1": {},
        }
        "intermediate": {
            "dense": {},
        },Nones,
        "output": {
            "dense": {},
        },
    }
    "linear_default": {},
    "matmul_default": {},
    "model_layer_0": {
        "attention": {
            ...
        },
        ...
    }
}
"""


def create_a_layer_config(
    linear_qc: dict = None, matmul_qc: dict = None, layer_qc=None
) -> dict:
    if (layer_qc is None and matmul_qc is None) and layer_qc is None:
        raise ValueError("Must provide either (linear_qc & matmul_qc) or layer_qc")

    if layer_qc is None:
        layer_qc = {}

    # fmt: off
    qc = {
        "attention": {
            "query": deepcopy(parse_node_config(layer_qc.get("attention", {}).get("query", linear_qc), "linear")),
            "key": deepcopy(parse_node_config(layer_qc.get("attention", {}).get("key", linear_qc), "linear")),
            "value": deepcopy(parse_node_config(layer_qc.get("attention", {}).get("value", linear_qc), "linear")),
            "matmul_0": deepcopy(parse_node_config(layer_qc.get("attention", {}).get("matmul_0", matmul_qc), "matmul")),
            "matmul_1": deepcopy(parse_node_config(layer_qc.get("attention", {}).get("matmul_1", matmul_qc), "matmul")),
            "output": {
                "dense": deepcopy(parse_node_config(layer_qc.get("attention", {}).get("output", {}).get("dense", linear_qc), "linear")),
            },
        },
        "intermediate": {
            "dense": deepcopy(parse_node_config(layer_qc.get("intermediate", {}).get("dense", linear_qc), "linear")),
        },
        "output": {
            "dense": deepcopy(parse_node_config(layer_qc.get("output", {}).get("dense", linear_qc), "linear")),
        },
    }
    # fmt: on
    return qc


def _parse_and_complete_config(
    config: dict,
    num_hidden_layers: int,
) -> dict:
    assert "default" in config, "Must provide a default config"
    default_qc: dict = config["default"]
    linear_qc: dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    matmul_qc: dict = parse_node_config(
        config.get("matmul", default_qc), mase_op="matmul"
    )
    general_layer_qc: dict = config.get("model_layer", None)

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_qc = config.get(layer_entry, general_layer_qc)
        p_config[layer_entry] = create_a_layer_config(linear_qc, matmul_qc, layer_qc)
    p_config["default"] = default_qc
    return p_config


def parse_bert_quantized_config(
    config: str | dict | None, num_hidden_layers: int
) -> dict:
    assert isinstance(
        config, (str, dict, type(None))
    ), "Must provide either a path, None or a dict"
    if config is None:
        return None
    if isinstance(config, str):
        config = toml.load(config)
    config = convert_str_na_to_none(config)
    parsed_config = _parse_and_complete_config(config, num_hidden_layers)
    return parsed_config
