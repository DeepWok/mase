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


from quantize_modules import MxIntPatchEmbed, VIT_CUSTOM_OPS

# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
# verified test case linear(2,4)


class CustomModel(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self,         
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        q_config: dict = None,
        norm_layer: nn.Module = nn.LayerNorm
        ) -> None:
        super().__init__()
        self.MxIntPatchEmbed = MxIntPatchEmbed(
            img_size, patch_size, in_chans, embed_dim, q_config, norm_layer
        )


    def forward(self, x):
        x = self.MxIntPatchEmbed(x)
        return x

quan_args = {
    "mx_int_patch_embed": {
        "config": {
        "name": "mxint_hardware",
        "data_in_width": 4,
        "data_in_exponent_width": 8,
        "data_in_parallelism": [3, 1, 1],

        "weight_width": 4,
        "weight_exponent_width": 8,
        "weight_parallelism": [32, 3],

        "bias_width": 4,
        "bias_exponent_width": 8,
        "bias_parallelism": [1, 32],

        "data_out_width": 4,
        "data_out_exponent_width": 8,
        "data_out_parallelism": [1, 32],
        }
    },
}

@pytest.mark.dev
def test_emit_verilog_linear():
    img_size = 224
    patch_size = 16
    embed_dim = 192
    in_chans = 3

    model_config = {
    }
    layer = CustomModel(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        q_config=quan_args,
    )



    mg = chop.MaseGraph(model=layer, custom_ops=VIT_CUSTOM_OPS)
    mg.model.custom_ops = VIT_CUSTOM_OPS
    # we have to have this batch size in advance
    x = torch.randn((1, in_chans, img_size, img_size))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    mg, _ = passes.graph.transforms.insert_fork_transform_pass(mg, quan_args)
    update_common_metadata_pass(mg, quan_args)
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    # model_config.pop("dim")
    model_args = {"vit_self_attention_integer": model_config}
    
    from functools import partial
    update_hardware_precision_param(mg, quan_args, model_args)
    updating_hardware_metadata_pass(mg, {
        "updating_funcs_list": [
            updating_for_patch_embed,
            ],
            }) 
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    pass_args = {
        "project_dir": Path("/scratch/cx922/mase/mxint_patch_embed"),
    }
    mg, _ = passes.emit_verilog_top_transform_pass(mg, pass_args)
    mg, _ = passes.emit_bram_transform_pass(mg, pass_args)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg, pass_args)
    mg, _ = passes.emit_vivado_project_transform_pass(mg, pass_args)


def _simulate():
    simulate(
        skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False
    )

def updating_for_patch_embed(node):
    mase_op = node.meta["mase"].parameters["common"]["mase_op"] 
    vp = node.meta["mase"]["hardware"].get("verilog_param")
    if mase_op == "mx_int_patch_embed":
        vp["DATA_IN_0_PARALLELISM_DIM_2"] = vp["DATA_IN_0_TENSOR_SIZE_DIM_2"]
        del vp["CLS_TOKEN_TENSOR_SIZE_DIM_2"]
        del vp["CLS_TOKEN_PARALLELISM_DIM_2"]
        del vp["DISTILL_TOKEN_TENSOR_SIZE_DIM_2"]
        del vp["DISTILL_TOKEN_PARALLELISM_DIM_2"]
        for dim in ["CONV_WEIGHT_PARALLELISM", "CONV_WEIGHT_TENSOR_SIZE"]:
            dim_0 = f"{dim}_DIM_0"
            dim_1 = f"{dim}_DIM_1"
            dim_2 = f"{dim}_DIM_2"
            dim_3 = f"{dim}_DIM_3"
            if dim_0 in vp and dim_1 in vp and dim_2 in vp and dim_3 in vp:
                vp[dim_0] = vp[dim_0] * vp[dim_1] * vp[dim_2]
                vp[dim_1] = vp[dim_3]
                del vp[dim_2]
                del vp[dim_3]
            else:
                raise ValueError(f"Cannot find {dim} in {vp}")

    
def updating_hardware_metadata_pass(mg, pass_args):
    for node in mg.fx_graph.nodes: 
        for func in pass_args["updating_funcs_list"]:
            node = func(node)
    
            
        

if __name__ == "__main__":
    test_emit_verilog_linear()
    # _simulate()
