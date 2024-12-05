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

def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


logger = get_logger(__name__)
sys.excepthook = excepthook

from chop.nn.quantized.modules.attention import _ViTAttentionBase

# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
# verified test case linear(2,4)

from quantize_modules import MxIntAddition, MxIntLinear, MxIntGELU, MxIntLayerNorm, ViTAttentionMxInt, vit_module_level_quantize

class MxIntBlock(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self, dim, num_heads) -> None:
        super().__init__()

        self.linear1 = nn.Linear(dim, 4*dim)
        self.act = torch.nn.GELU()
        self.linear2 = nn.Linear(4*dim, dim)

        self.add = MxIntAddition({})
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

class MxIntStreamBlocks(torch.nn.Module):

    def __init__(self, dim, num_heads, stream_depth) -> None:
        super().__init__()
        
        self.stream_depth = stream_depth
        self.stream_blocks = nn.Sequential(
            *[
                MxIntBlock(
                    dim=dim,
                    num_heads=num_heads,
                )
                for i in range(stream_depth)
            ]
        )
    def forward(self, x):
        return self.stream_blocks(x)

class MxIntFoldedTopBlocks(torch.nn.Module):

    def __init__(self, dim, num_heads, stream_depth, folded_depth) -> None:
        super().__init__()
        
        self.folded_depth = folded_depth
        self.folded_blocks = nn.Sequential(
            *[
                MxIntStreamBlocks(
                    dim=dim,
                    num_heads=num_heads,
                    stream_depth=stream_depth
                )
                for i in range(folded_depth)
            ]
        )
    def forward(self, x):
        return self.folded_blocks(x)

quan_args = {
    "layer_norm": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 4,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, 32],

        "weight_width": 4,
        "weight_exponent_width": 8,
        "weight_parallelism": [1, 32],

        "bias_width": 4,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, 32],

        "data_out_width": 4,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, 32],
        }
    },
    "gelu": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 4,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, 32],

        "weight_width": 4,
        "weight_exponent_width": 8,
        "weight_parallelism": [32, 32],

        "bias_width": 4,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, 32],

        "data_out_width": 4,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, 32],
        },
    },
    "linear": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 4,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, 32],

        "weight_width": 4,
        "weight_exponent_width": 8,
        "weight_parallelism": [32, 32],

        "bias_width": 4,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, 32],

        "data_out_width": 4,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, 32],
        },
    },
    "user_defined_module": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 4,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, 32],

        "weight_width": 4,
        "weight_exponent_width": 8,
        "weight_parallelism": [32, 32],

        "bias_width": 4,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, 32],

        "data_out_width": 4,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, 32],
        }
    },
    "fork2": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 4,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [1, 32],
        "data_out_width": 4,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, 32],
        }
    },
}
attention_quant_config = {
    "name": "mxint_hardware",
    "data_in_width": 4,
    "data_in_exponent_width": 8,
    "data_in_parallelism": [1, 32],

    "weight_width": 4,
    "weight_exponent_width": 8,
    "weight_parallelism": [32, 32],

    "bias_width": 4,
    "bias_exponent_width": 8,
    "bias_parallelism": [1, 32],

    "data_out_width": 4,
    "data_out_exponent_width": 8,
    "data_out_parallelism": [1, 32],
}

from mase_components import get_module_dependencies
VIT_CUSTOM_OPS = {
    "modules": {
        ViTAttentionMxInt: {
            "args": {
                "dim": "data_in",
                "num_heads": "config",
                "qkv_bias": "config",
                "qk_norm": None,
                "attn_drop": None,
                "proj_drop": None,
                "norm_layer": None,
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_vit_attention_wrap",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_vit_attention_wrap"
            ),
        },
        MxIntLayerNorm: {
            "args": {
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_layernorm",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_layernorm"
            ),
        },
        MxIntGELU: {
            "args": {
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_gelu",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_gelu"
            ),
        },
        MxIntLinear: {
            "args": {
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_linear",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_linear"
            ),
        },
        MxIntAddition: {
            "args": {
                "input_0": "data_in",
                "input_1": "data_in",
                "q_config": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "mxint_addition",
            "dependence_files": get_module_dependencies(
                "linear_layers/mxint_operators/mxint_addition"
            ),
        },
    },
}


def graph_generation(model, pass_args):
    model_config = pass_args["model_config"]
    quan_args = pass_args["quan_args"]

    batch_size = pass_args["model_config"]["batch_size"]
    dim = pass_args["model_config"]["dim"]
    n = pass_args["model_config"]["n"]
    x = torch.randn((batch_size, n, dim))
    dummy_in = {"x": x}

    qmodel = vit_module_level_quantize(model, model_config, quan_args)
    mg = chop.MaseGraph(model=qmodel, custom_ops=VIT_CUSTOM_OPS)
    mg.model.custom_ops = VIT_CUSTOM_OPS
    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    mg, _ = passes.graph.transforms.insert_fork_transform_pass(mg, quan_args)
    update_common_metadata_pass(mg, quan_args)
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    model_args = {"vit_self_attention_integer": model_config}
    update_hardware_precision_param(mg, quan_args, model_args)
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    return mg, _ 
    

@pytest.mark.dev
def test_emit_verilog_linear():
    batch_size = 1
    n = 196
    dim = 192
    num_heads = 3
    # notice: 
    stream_depth = 2
    folded_depth = 6
    
    model_config = {
        "batch_size": batch_size,
        "n": n,
        "dim": dim,
        "num_heads": num_heads,
    }

    stream_layer = MxIntStreamBlocks(dim, num_heads, stream_depth=stream_depth)
    top_layer = MxIntFoldedTopBlocks(dim, num_heads, stream_depth=stream_depth, folded_depth=folded_depth)
    stream_mg, _ = graph_generation(stream_layer, {"model_config": model_config, "quan_args": quan_args})
    top_mg, _ = graph_generation(top_layer, {"model_config": model_config, "quan_args": quan_args})


    pass_args = {
        "project_dir": Path("/scratch/cx922/mase/mxint_vit_folded_top"),
    }

    from utils_mxint_folded_top_generation import mxint_folded_top_generation
    mxint_folded_top_generation(
        top_mg, 
        pass_args={
            "stream_graph": stream_mg,
            "stream_name": "stream_blocks",
            "folded_name": "folded_blocks",
            "reuse_times": folded_depth,
            "project_dir": Path("/scratch/cx922/mase/mxint_vit_folded_top")
        }
    )
    # top_mg, _ = passes.emit_bram_transform_pass(top_mg, pass_args)
    top_mg, _ = passes.emit_internal_rtl_transform_pass(top_mg, pass_args)
    top_mg, _ = passes.emit_vivado_project_transform_pass(top_mg, pass_args)

    # mg, _ = passes.emit_cocotb_transform_pass(
    #     mg, pass_args={"wait_time": 100, "wait_unit": "us", "batch_size": batch_size}
    # )
    top_mg, _ = passes.emit_vivado_project_transform_pass(top_mg)
def _simulate():
    simulate(
        skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False
    )


if __name__ == "__main__":
    test_emit_verilog_linear()
    # _simulate()
