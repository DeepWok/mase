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
from test_emit_verilog_layernorm import (
    update_common_metadata_pass,
    update_hardware_precision_param,
)
from chop.nn.quantized.modules.attention import ViTAttentionInteger
from mase_components import get_module_dependencies
from chop.models.vision.vit.vit import Attention

set_logging_verbosity("debug")


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


logger = get_logger(__name__)
sys.excepthook = excepthook
VIT_CUSTOM_OPS = {
    "modules": {
        ViTAttentionInteger: {
            "args": {
                "hidden_states": "data_in",
                "attention_mask": None,
                "head_mask": None,
                "encoder_hidden_states": None,
                "encoder_attention_mask": None,
                "past_key_value": None,
                "output_attentions": "config",
            },
            "toolchain": "INTERNAL_RTL",
            "module": "fixed_self_attention",
            "dependence_files": get_module_dependencies(
                "vision_models/vit/rtl/fixed_vit_attention_single_precision_wrapper"
            ),
        },
    }
}

# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
# verified test case linear(2,4)


class Layer(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self, dim, num_heads) -> None:
        super().__init__()
        self.attention = Attention(dim, num_heads, qkv_bias=True)

    def forward(self, x):
        x = self.attention(x)
        return x


attention_quant_config = {
    "name": "integer_floor",
    "data_in_width": 8,
    "data_in_frac_width": 4,
    "qkv_weight_width": 6,
    "qkv_weight_frac_width": 4,
    "qkv_bias_width": 6,
    "qkv_bias_frac_width": 4,
    "qkv_width": 8,
    "qkv_frac_width": 4,
    "qkmm_out_width": 8,
    "qkmm_out_frac_width": 5,
    "softmax_exp_width": 8,
    "softmax_exp_frac_width": 3,
    "softmax_out_frac_width": 9,
    "svmm_out_width": 8,
    "svmm_out_frac_width": 4,
    "proj_weight_width": 6,
    "proj_weight_frac_width": 4,
    "proj_bias_width": 8,
    "proj_bias_frac_width": 4,
    "data_out_width": 8,
    "data_out_frac_width": 4,
}

quan_args = {
    "by": "type",  # quantize by type, name, or regex_name
    "default": {
        "config": {"name": None}
    },  # default config, this would be used for any node that does not have a specific config
    "vit_self_attention_integer": {"config": attention_quant_config},
}


from chop.passes.graph.utils import deepsetattr


def vit_module_level_quantize(model, model_config, q_config):
    for module in model.named_modules():
        if isinstance(module[1], Attention):
            ori_module = module[1]
            new_module = ViTAttentionInteger(
                model_config["dim"],
                model_config["num_heads"],
                qkv_bias=model_config["query_has_bias"],
                q_config=q_config,
            )
            logger.info(f"Replacing module: {module[0]}")
            dim = ori_module.head_dim * ori_module.num_heads

            qkv_weight = ori_module.qkv.weight.reshape(3, dim, dim)
            new_module.query.weight = nn.Parameter(qkv_weight[0])
            new_module.key.weight = nn.Parameter(qkv_weight[1])
            new_module.value.weight = nn.Parameter(qkv_weight[2])

            has_bias = False if ori_module.qkv.bias == None else True
            if has_bias:
                qkv_bias = ori_module.qkv.bias.reshape(3, 1, dim)
                new_module.query.bias = nn.Parameter(qkv_bias[0])
                new_module.key.bias = nn.Parameter(qkv_bias[1])
                new_module.value.bias = nn.Parameter(qkv_bias[2])

            new_module.proj.weight = ori_module.proj.weight
            new_module.proj.bias = ori_module.proj.bias
            deepsetattr(model, module[0], new_module)
    return model


@pytest.mark.dev
def test_emit_verilog_vit_attention():
    dim = 40
    num_heads = 4
    batch_size = 1
    n = 20
    model_config = {
        "dim": dim,
        "num_heads": num_heads,
        "QUERY_WEIGHTS_PRE_TRANSPOSED": False,
        "query_has_bias": True,
    }
    layer = Layer(dim, num_heads)
    qlayer = vit_module_level_quantize(layer, model_config, attention_quant_config)
    mg = chop.MaseGraph(model=qlayer, custom_ops=VIT_CUSTOM_OPS)
    torch.manual_seed(0)
    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, n, dim))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    mg, _ = passes.quantize_transform_pass(mg, quan_args)
    update_common_metadata_pass(mg, quan_args)
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    model_config.pop("dim")
    model_args = {"vit_self_attention_integer": model_config}
    update_hardware_precision_param(mg, quan_args, model_args)
    # TODO:
    # Currently, the common metadata pass does not support extracting extra arguments except data in.
    # So we need to have it directly after adding hardware metadata
    mg, _ = passes.report_node_hardware_type_analysis_pass(
        mg,
        pass_args={
            "which": ["common", "hardware"],
            "save_path": "graph_meta_params.txt",
        },
    )  # pretty print
    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg, pass_args={"wait_time": 10, "wait_units": "ms", "batch_size": batch_size}
    )
    # mg, _ = passes.emit_vivado_project_transform_pass(mg)

    simulate(
        skip_build=False, skip_test=False, simulator="questa", trace_depth=5, waves=True
    )


if __name__ == "__main__":
    test_emit_verilog_vit_attention()
