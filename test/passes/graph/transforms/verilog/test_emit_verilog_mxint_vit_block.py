#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os, sys, logging, traceback, pdb
import pytest
import toml

import torch
import torch.nn as nn

import chop as chop
import chop.passes as passes

from pathlib import Path

from chop.actions import simulate
from chop.tools.logger import set_logging_verbosity
from chop.tools import get_logger

set_logging_verbosity("debug")
from utils import (
    update_common_metadata_pass,
    update_hardware_precision_param,
    manually_update_hardware_parallelism_param,
)
from chop.models.vision.vit.vit import Attention

from mxint_quant import (
    vit_module_level_quantize, 
    MXIntAddition, 
    VIT_CUSTOM_OPS
)
def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)
logger = get_logger(__name__)
sys.excepthook = excepthook

class CustomModel(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self, dim, num_heads) -> None:
        super().__init__()

        # self.linear = nn.Linear(dim, dim)
        self.linear1 = nn.Linear(dim, 4*dim)
        self.act = torch.nn.GELU()
        self.linear2 = nn.Linear(4*dim, dim)

        self.add = MXIntAddition({})
        self.norm1 = torch.nn.LayerNorm(dim)

        self.attention = Attention(dim, num_heads, qkv_bias=True)
        self.norm2 = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.act(x1)
        x1 = self.linear2(x1)
        x1 = self.norm1(x1)
        
        mlp = self.add(x1, x)
        
        attn = self.attention(mlp)
        attn = self.norm2(attn)
        result = self.add(attn, mlp)
        return result
parallelism = 16
quan_args = {
    "by": "type",
    "layer_norm": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 8,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, parallelism],

        "weight_width": 6,
        "weight_exponent_width": 8,
        "weight_parallelism": [1, parallelism],

        "bias_width": 6,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, parallelism],

        "data_out_width": 8,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, parallelism],
        }
    },
    "gelu": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 8,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, parallelism],

        "weight_width": 6,
        "weight_exponent_width": 8,
        "weight_parallelism": [parallelism, parallelism],

        "bias_width": 6,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, parallelism],

        "data_out_width": 8,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, parallelism],
        },
    },
    "linear": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 8,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, parallelism],

        "weight_width": 6,
        "weight_exponent_width": 8,
        "weight_parallelism": [parallelism, parallelism],

        "bias_width": 6,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, parallelism],

        "data_out_width": 8,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, parallelism],
        },
    },
    "attention": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 8,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, parallelism],

        "weight_width": 6,
        "weight_exponent_width": 8,
        "weight_parallelism": [parallelism, parallelism],

        "bias_width": 6,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, parallelism],

        "data_out_width": 8,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, parallelism],
        }
    },
    "fork2": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 8,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, parallelism],
        "data_out_width": 8,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, parallelism],
        }
    },
    "add": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 8,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, parallelism],
        "data_out_width": 8,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, parallelism],
        }
    },
}

@pytest.mark.dev
def test_emit_verilog_linear():
    dim = 64
    num_heads = 2
    batch_size = 1
    n = 2
    model_config = {
        "dim": dim,
        "num_heads": num_heads,
        "query_has_bias": True,
    }
    layer = CustomModel(dim, num_heads)
    qlayer = vit_module_level_quantize(layer, model_config, quan_args)
    mg = chop.MaseGraph(model=qlayer, custom_ops=VIT_CUSTOM_OPS)
    mg.model.custom_ops = VIT_CUSTOM_OPS
    # torch.manual_seed(0)
    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, n, dim))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    mg, _ = passes.graph.transforms.insert_fork_transform_pass(mg, quan_args)
    update_common_metadata_pass(mg, quan_args)
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    model_config.pop("dim")
    model_args = {"vit_self_attention_integer": model_config}
    update_hardware_precision_param(mg, quan_args, model_args)
    from utils import updating_hardware_metadata_pass
    updating_hardware_metadata_pass(mg, {"updating_funcs_list": [del_layernorm_args]})
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    pass_args = {
        "project_dir": Path("mxint_vit_block"),
    }
    mg, _ = passes.emit_verilog_top_transform_pass(mg, pass_args)
    mg, _ = passes.emit_bram_transform_pass(mg, pass_args)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg, pass_args)
    mg, _ = passes.emit_vivado_project_transform_pass(mg, pass_args)


def del_layernorm_args(node):
    print(node.name)
    if "layernorm" in node.name:
        del node.meta["mase"].parameters["hardware"]["verilog_param"]["NORMALIZED_SHAPE"]
        del node.meta["mase"].parameters["hardware"]["verilog_param"]["EPS"]


def _simulate():
    simulate(
        skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False
    )


if __name__ == "__main__":
    test_emit_verilog_linear()
    # _simulate()
