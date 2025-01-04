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
class ViTAttentionMxInt(_ViTAttentionBase):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        q_config: dict = None,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop)
        self.q_config = q_config

def vit_module_level_quantize(model, model_config, q_config):
    from chop.passes.graph.utils import deepsetattr
    for module in model.named_modules():
        if isinstance(module[1], Attention):
            ori_module = module[1]
            new_module = ViTAttentionMxInt(
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

# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
# verified test case linear(2,4)


class CustomModel(torch.nn.Module):
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
    "name": "mxint_hardware",
    "data_in_width": 4,
    "data_in_exponent_width": 8,
    "data_in_parallelism": [1, 48],

    "weight_width": 4,
    "weight_exponent_width": 8,
    "weight_parallelism": [48, 48],

    "bias_width": 4,
    "bias_exponent_width": 8,
    "bias_parallelism": [1, 48],

    "data_out_width": 4,
    "data_out_exponent_width": 8,
    "data_out_parallelism": [1, 48],
}

quan_args = {
    "by": "type",  # quantize by type, name, or regex_name
    "default": {
        "config": {"name": None}
    },  # default config, this would be used for any node that does not have a specific config
    "user_defined_module": {"config": attention_quant_config},
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
    },
}
@pytest.mark.dev
def test_emit_verilog_linear():
    dim = 192
    num_heads = 3
    batch_size = 1
    n = 196
    model_config = {
        "dim": dim,
        "num_heads": num_heads,
        "query_has_bias": True,
    }
    layer = CustomModel(dim, num_heads)
    qlayer = vit_module_level_quantize(layer, model_config, attention_quant_config)
    mg = chop.MaseGraph(model=qlayer, custom_ops=VIT_CUSTOM_OPS)
    mg.model.custom_ops = VIT_CUSTOM_OPS
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
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [2] * 4}
    )
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print
    pass_args = {
        "project_dir": Path("/scratch/cx922/mase/mxint_vit_attention"),
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


if __name__ == "__main__":
    test_emit_verilog_linear()
    # _simulate()
