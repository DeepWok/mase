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
sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from a_cx_mxint_quant import (
    vit_module_level_quantize, 
    VIT_CUSTOM_OPS
)

def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


logger = get_logger(__name__)
sys.excepthook = excepthook


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
# verified test case linear(2,4)

class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self, in_features, hidden_features, out_features) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = torch.nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

parallelism = 64
parallelism2 = 12
quan_args = {
    "by": "type",  # quantize by type, name, or regex_name
    "default": {
        "config": {"name": None}
    },  # default config, this would be used for any node that does not have a specific config
    "linear": {
        "config": {
            "name": "mxint_hardware",
            # data
            "data_in_width": 6,
            "data_in_exponent_width": 4,
            "data_in_parallelism": [1, parallelism],
            # weight
            "weight_width": 6,
            "weight_exponent_width": 4,
            "weight_parallelism": [parallelism, parallelism],
            # bias
            "bias_width": 6,
            "bias_exponent_width": 4,
            "bias_parallelism": [1, parallelism],
            "data_out_width": 6,
            "data_out_exponent_width": 4,
            "data_out_parallelism": [1, parallelism],
            "round_bits": 4,
        }
    },
    "gelu": {
        "config": {
            "name": "mxint_hardware",
            "data_in_width": 6,
            "data_in_exponent_width": 4,
            "data_in_parallelism": [1, parallelism],
            "hash_out_width": 10,
            "data_out_width": 6,
            "data_out_exponent_width": 4,
            "data_out_parallelism": [1, parallelism],
        }
    }
}

@pytest.mark.dev
def test_emit_verilog_linear():
    in_features = 192
    hidden_features = 192*4
    out_features = 192
    n = 196
    batch_size = 10
    layer = MLP(in_features, hidden_features, out_features)
    qlayer = vit_module_level_quantize(layer, q_config=quan_args)
    model_path = "/scratch/cx922/mase/mlp_model.pth"
    torch.save(qlayer.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    mg = chop.MaseGraph(model=qlayer, custom_ops=VIT_CUSTOM_OPS)
    mg.model.custom_ops = VIT_CUSTOM_OPS
    torch.manual_seed(0)
    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, n, in_features))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    update_common_metadata_pass(mg, quan_args)
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    update_hardware_precision_param(mg, quan_args)

    from utils import updating_hardware_metadata_pass
    from functools import partial
    updating_hardware_metadata_pass(mg, {
        "updating_funcs_list": [
            partial(updating_for_mlp, quan_args=quan_args),
            ],
            }) 
    
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    pass_args = {
        "project_dir": Path("/scratch/cx922/mase/mxint_mlp"),
    }
    mg, _ = passes.emit_verilog_top_transform_pass(mg, pass_args)
    mg, _ = passes.emit_bram_transform_pass(mg, pass_args)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg, pass_args)
    # mg, _ = passes.emit_cocotb_transform_pass(
    #     mg, pass_args={"wait_time": 100, "wait_unit": "ms", "batch_size": batch_size}
    # )
    mg, _ = passes.emit_vivado_project_transform_pass(mg, pass_args)


def _simulate():
    simulate(
        skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False
    )
def updating_for_mlp(node, quan_args):
    mase_op = node.meta["mase"].parameters["common"]["mase_op"] 
    vp = node.meta["mase"]["hardware"].get("verilog_param")
    if mase_op == "gelu":
        vp["HASH_OUT_WIDTH"] = quan_args["gelu"]["config"]["hash_out_width"]

if __name__ == "__main__":
    test_emit_verilog_linear()
    # _simulate()
