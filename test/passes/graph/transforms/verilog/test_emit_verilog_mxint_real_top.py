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

sys.path.append(Path(__file__).resolve().parents[5].as_posix())
logger = get_logger(__name__)
sys.excepthook = excepthook


from a_cx_mxint_quant.modules import MXIntPatchEmbed
from a_cx_mxint_quant import VIT_CUSTOM_OPS

# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
# verified test case linear(2,4)


from a_cx_mxint_quant import MXIntLinear, MXIntGELU

class MXIntFoldedTop(torch.nn.Module):
    def __init__(self, q_config):
        super().__init__()
        self.q_config = q_config
    def forward(self, x):
        return x

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
        self.MXIntPatchEmbed = MXIntPatchEmbed(
            img_size, patch_size, in_chans, embed_dim, q_config, norm_layer
        )
        self.folded_block = MXIntGELU(q_config=q_config["folded_block"]["config"])
        self.head = MXIntLinear(
            embed_dim, 10, q_config=q_config["head"]["config"]
        )
    def forward(self, x):
        x = self.MXIntPatchEmbed(x)
        x = self.folded_block(x)
        x = self.head(x)
        return x

def get_parallelism(config, parallelism, mlp_parallelism):
    quan_args = {
        "by": "name",
        "mxint_patch_embed": {
            "config": {
            "name": "mxint_hardware",
            "data_in_width": config["data_width"],
            "data_in_exponent_width": config["data_exponent_width"],
            "data_in_parallelism": [3, 1, 1],

            "weight_width": config["weight_width"],
            "weight_exponent_width": config["weight_exponent_width"],
            "weight_parallelism": [parallelism, 3, 1, 1],

            "bias_width": config["bias_width"],
            "bias_exponent_width": config["bias_exponent_width"],
            "bias_parallelism": [1, parallelism],

            "data_out_width": config["data_width"],
            "data_out_exponent_width": config["data_exponent_width"],
            "data_out_parallelism": [1, parallelism],
            }
        },
        "folded_block": {
            "config": {
            "name": "mxint_hardware",
            "data_in_width": config["data_width"],
            "data_in_exponent_width": config["data_exponent_width"],
            "data_in_parallelism": [1, parallelism],

            "hash_out_width": 5,

            "data_out_width": config["data_width"],
            "data_out_exponent_width": config["data_exponent_width"],
            "data_out_parallelism": [1, parallelism],
            }
        },
        "head": {
            "config": {
            "name": "mxint_hardware",
            "data_in_width": config["data_width"],
            "data_in_exponent_width": config["data_exponent_width"],
            "data_in_parallelism": [1, parallelism],

            "weight_width": config["weight_width"],
            "weight_exponent_width": config["weight_exponent_width"],
            "weight_parallelism": [1, parallelism],

            "bias_width": config["bias_width"],
            "bias_exponent_width": config["bias_exponent_width"],
            "bias_parallelism": [1, 1],

            "data_out_width": config["data_width"],
            "data_out_exponent_width": config["data_exponent_width"],
            "data_out_parallelism": [1, 1],
            }
        },
    }
    return quan_args

@pytest.mark.dev
def test_emit_verilog_linear():
    import yaml
    config_path = os.environ.get("CONFIG_PATH")
    args = yaml.safe_load(open(config_path))
    config = args["config"]
    parallelism = args["parallelism"]
    mlp_parallelism = args["mlp_parallelism"]
    quan_args = get_parallelism(config, parallelism, mlp_parallelism)

    img_size = int(args["img_size"])
    patch_size = int(args["patch_size"])
    embed_dim = int(args["embed_dim"])
    in_chans = int(args["in_chans"])
    project_dir = Path(args["project_dir"])


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
    model_args = {"vit_self_attention_integer": {}}
    
    from functools import partial
    from utils import updating_hardware_metadata_pass
    update_hardware_precision_param(mg, quan_args, model_args)
    updating_hardware_metadata_pass(mg, {
        "updating_funcs_list": [
            updating_for_patch_embed,
            ],
            }) 
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    pass_args = {
        "project_dir": Path(project_dir),
        "real_top": True,
    }
    mg, _ = passes.emit_verilog_top_transform_pass(mg, pass_args)
    mg, _ = passes.emit_bram_transform_pass(mg, pass_args)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg, pass_args)
    with open(project_dir / "config.yaml", "w") as f:
        yaml.dump(args, f)
    # mg, _ = passes.emit_vivado_project_transform_pass(mg, pass_args)


def _simulate():
    simulate(
        skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False
    )

def updating_for_patch_embed(node):
    mase_op = node.meta["mase"].parameters["common"]["mase_op"] 
    vp = node.meta["mase"]["hardware"].get("verilog_param")
    if mase_op == "mxint_patch_embed":
        vp["DATA_IN_0_TENSOR_SIZE_DIM_2"] = 3
        vp["DATA_IN_0_PARALLELISM_DIM_2"] = 3
        del vp["DATA_IN_0_TENSOR_SIZE_DIM_3"]
        del vp["DATA_IN_0_PARALLELISM_DIM_3"]
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
        vp["CLS_TOKEN_PRECISION_0"] = vp["DATA_IN_0_PRECISION_0"]
        vp["CLS_TOKEN_PRECISION_1"] = vp["DATA_IN_0_PRECISION_1"]
        vp["DISTILL_TOKEN_PRECISION_0"] = vp["DATA_IN_0_PRECISION_0"]
        vp["DISTILL_TOKEN_PRECISION_1"] = vp["DATA_IN_0_PRECISION_1"]

    

    
            
        

if __name__ == "__main__":
    test_emit_verilog_linear()
    # _simulate()
