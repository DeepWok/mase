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
from chop.passes.graph.transforms.quantize import QUANTIZEABLE_OP
set_logging_verbosity("debug")


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


logger = get_logger(__name__)
sys.excepthook = excepthook

torch.manual_seed(0)
# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
# verified test case linear(2,4)

class LAYERNORM_MODULE(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self, norm_dim) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(norm_dim,elementwise_affine=True)
        if self.norm.elementwise_affine:
            self.norm.weight = torch.nn.Parameter(torch.rand(norm_dim))
            if self.norm.bias is not None:
                self.norm.bias = torch.nn.Parameter(torch.rand(norm_dim))

    def forward(self, x):
        x = self.norm(x)
        return x

def update_common_metadata_pass(
    mg, quan_args
):
    # There is a bug in the current quantization pass, where the results metadata is not updated with the precision.
    # # Here we update the metadata here so we can test the hardware back end.
    for node in mg.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        if mase_op not in QUANTIZEABLE_OP:
            print(mase_op)
            continue
        node_quan_config = quan_args.get(mase_op)["config"]
        for arg, _ in node.meta["mase"].parameters["common"]["args"].items():
            if (
                type(node.meta["mase"].parameters["common"]["args"][arg]) == dict
                and "type" in node.meta["mase"].parameters["common"]["args"][arg].keys()
            ):
                node.meta["mase"].parameters["common"]["args"][arg]["type"] = "fixed"
        for result, _ in node.meta["mase"].parameters["common"]["results"].items():
            if (
                type(node.meta["mase"].parameters["common"]["results"][result]) == dict
                and "type"
                in node.meta["mase"].parameters["common"]["results"][result].keys()
            ):
                node.meta["mase"].parameters["common"]["results"][result][
                    "type"
                ] = "fixed"
                node.meta["mase"].parameters["common"]["results"][result][
                    "precision"
                ] = [node_quan_config["data_out_width"], node_quan_config["data_out_frac_width"]]
    
    for node in mg.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        if mase_op in ["layer_norm"]:
            if node.meta["mase"].parameters["common"]["args"].get("weight")!=None:
                node.meta["mase"].parameters["common"]["args"]["elementwise_affine"] = True
                if node.meta["mase"].parameters["common"]["args"].get("bias") !=None:
                    node.meta["mase"].parameters["common"]["args"]["has_bias"] = True
        
def update_hardware_precision_param(
    mg, quan_args
):
    # The quantization pass currently don't support any inlayer precision automatically generate
    # we only have data_in, weight.. param in common metadata
    # in order to support in layer fine grained precision tuning
    # we just update the hardware metadata directly.
    def _cap(name):
        """
        capitalize a string
        """
        return str(name).upper()
    for node in mg.fx_graph.nodes:
        mase_op = node.meta["mase"].parameters["common"]["mase_op"]
        if mase_op not in QUANTIZEABLE_OP:
            continue
        vp = node.meta["mase"]["hardware"]["verilog_param"]
        node_quan_config = quan_args.get(mase_op)["config"]
        if mase_op not in ["layer_norm"]:
            continue
        for config_name, config_info in node_quan_config.items():
            _list = ["data_in","data_out","weight","bias"]
            if any(keyword in config_name for keyword in ["data_in", "data_out", "weight", "bias"]):
                continue
            if "width" not in config_name:
                continue
            cofig_str = config_name.replace("frac_width", "precision_1")
            cofig_str = cofig_str.replace("width", "precision_0")
            vp[_cap(cofig_str)] = config_info

quan_args = {
    "by": "type", # quantize by type, name, or regex_name
    "default": {"config": {"name": None}}, # default config, this would be used for any node that does not have a specific config
    "layer_norm": {
        "config": {
            "name": "integer_floor", 
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
            "isqrt_in_width": 8,
            "isqrt_in_frac_width": 4,
            "isqrt_out_width": 8,
            "isqrt_out_frac_width": 4,
            "data_out_width": 8,
            "data_out_frac_width": 4,
            "bypass": False,
            "noparse": True,
        }
    }
}
@pytest.mark.dev
def test_emit_verilog_layernorm():
    
    batch_size = 4
    norm_dim = 8
    norm_layer = LAYERNORM_MODULE(norm_dim)
    mg = chop.MaseGraph(model=norm_layer)
    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, norm_dim))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    mg, _ = passes.quantize_transform_pass(mg, quan_args)
    update_common_metadata_pass(mg,quan_args)
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    update_hardware_precision_param(mg,quan_args)
    print(mg.meta["mase"]["common"]["args"])
    # mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg, pass_args={"wait_time": 100, "wait_unit": "ms", "batch_size": batch_size}
    )
    # mg, _ = passes.emit_vivado_project_transform_pass(mg)

    simulate(skip_build=False, skip_test=False, simulator="verilator", waves=True)


if __name__ == "__main__":
    test_emit_verilog_layernorm()
