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


# def format_stat_profiled_int_config_bert_quantized(
#     config: dict,
#     num_hidden_layers: int,
#     default_config: dict = None,
#     is_ptq: bool = False,
#     bypass: bool = False,
# ):
#     """
#     nn.Module forward hook cannot be used to collect the statistics of torch functions (bmm, matmul)
#     Thus a hack is to collect the previous nn.Module's output

#     This formatter converts the previous nn.Module's output to the current torch function's input quant config
#     """

#     if default_config is None:
#         default_config = {
#             "name": "integer",
#             "bypass": bypass,
#             "is_ptq": is_ptq,
#             "data_in_width": 8,
#             "data_in_frac_width": 4,
#             "weight_width": 8,
#             "weight_frac_width": 8,
#             "bias_width": 8,
#             "bias_frac_width": 8,
#         }

#     for i in range(num_hidden_layers):
#         layer_entry = f"model_layer_{i}"
#         if layer_entry not in config:
#             raise ValueError(f"layer_entry {layer_entry} not found in config")

#         layer_config = config[layer_entry]

#         # fmt: off
#         layer_config["attention"]["matmul_0"] = {
#             "name": "integer",
#             "bypass": bypass,
#             "is_ptq": is_ptq,
#             "data_in_width": layer_config["attention"]["query"]["data_out_width"],
#             "data_in_frac_width": layer_config["attention"]["query"]["data_out_frac_width"],
#             "weight_width": layer_config["attention"]["key"]["data_out_width"],
#             "weight_frac_width": layer_config["attention"]["key"]["data_out_frac_width"],
#         }

#         try:
#             matmul_1_x_width = default_config[layer_entry]["attention"]["matmul_1"]["data_in_width"]
#             logger.debug("matmul_1 weight_width uses default_config[\"model_layer_x\"][\"attention\"][\"matmul_1\"][\"data_in_width\"]")
#         except KeyError:
#             matmul_1_x_width = default_config["data_in_width"]
#             logger.debug("matmul_1 weight_width default_config[\"data_in_width\"]")

#         layer_config["attention"]["matmul_1"] = {
#             "name": "integer",
#             "bypass": bypass,
#             "is_ptq": is_ptq,
#             "data_in_width": matmul_1_x_width,
#             "data_in_frac_width": matmul_1_x_width-1,
#             "weight_width": layer_config["attention"]["value"]["data_out_width"],
#             "weight_frac_width": layer_config["attention"]["value"]["data_out_frac_width"],
#         }
#         # fmt: on

#         layer_config["attention"]["query"].pop("data_out_width")
#         layer_config["attention"]["query"].pop("data_out_frac_width")
#         layer_config["attention"]["key"].pop("data_out_width")
#         layer_config["attention"]["key"].pop("data_out_frac_width")
#         layer_config["attention"]["value"].pop("data_out_width")
#         layer_config["attention"]["value"].pop("data_out_frac_width")

#     if "default" not in config:
#         config["default"] = default_config.get(
#             "default",
#             {
#                 "name": "integer",
#                 "bypass": bypass,
#                 "is_ptq": is_ptq,
#                 "data_in_width": 8,
#                 "data_in_frac_width": 4,
#                 "weight_width": 8,
#                 "weight_frac_width": 8,
#                 "bias_width": 8,
#                 "bias_frac_width": 8,
#             },
#         )
#     return config
