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

set_logging_verbosity("debug")


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
from chop.models.vision.vit.vit import Attention

from chop.nn.quantized.modules.attention import ViTAttentionInteger
from mase_components import get_module_dependencies

VIT_CUSTOM_OPS = {"modules": {ViTAttentionInteger: {}}}


class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self, in_features, hidden_features) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x)))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = MLP,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # x = self.attn(x)
        return x


class ViTAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        depth: int = 12,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = MLP,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


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
    "softmax_out_frac_width": 7,
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
    "fork2": [8, 4],
    "linear": {
        "config": {
            "name": "integer_floor",  # quantization scheme name supported are ["integer", "fixed" (equivalent to integer), "lutnet" (dev mode), "logicnets" (dev mode), "binary", "binary_residual", "ternary", "minifloat_ieee", "minifloat_denorm", "log", "block_fp", "block_minifloat", "block_log"]
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 10,
            "weight_frac_width": 3,
            # bias
            "bias_width": 5,
            "bias_frac_width": 2,
            # optional
            "data_out_width": 8,
            "data_out_frac_width": 4,
        },
    },
    "gelu": {
        "config": {
            "name": "integer_floor",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "data_out_width": 8,
            "data_out_frac_width": 4,
        }
    },
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
            "isqrt_in_frac_width": 3,
            "isqrt_out_width": 8,
            "isqrt_out_frac_width": 7,
            "data_out_width": 8,
            "data_out_frac_width": 4,
            "bypass": False,
            "noparse": True,
        }
    },
    "add": {
        "config": {
            "name": "integer_floor",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "data_out_width": 8,
            "data_out_frac_width": 4,
        },
    },
    "vit_self_attention_integer": {"config": attention_quant_config},
}


@pytest.mark.dev
def test_emit_verilog_vit():
    # vit_tiny dim 192, n 196, num_heads = 3
    #
    dim = 192
    num_heads = 3
    batch_size = 1
    n = 196
    layer = ViTAttention(dim, num_heads, mlp_ratio=4, qkv_bias=True, depth=1)
    model_config_for_quantize = {
        "dim": dim,
        "num_heads": num_heads,
        "query_has_bias": True,
    }
    model_args_for_hardware_param = {
        "vit_self_attention_integer": {
            "num_heads": num_heads,
            "query_has_bias": True,
        }
    }
    qlayer = vit_module_level_quantize(
        layer, model_config_for_quantize, attention_quant_config
    )
    mg = chop.MaseGraph(model=qlayer)
    torch.manual_seed(0)
    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, n, dim))
    dummy_in = {"x": x}
    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    mg, _ = passes.quantize_transform_pass(mg, quan_args)
    mg, _ = passes.graph.transforms.insert_fork_transform_pass(mg, quan_args)
    update_common_metadata_pass(mg, quan_args)
    mg, _ = passes.add_hardware_metadata_analysis_pass(
        mg, pass_args={"max_parallelism": [1] * 4}
    )
    update_hardware_precision_param(mg, quan_args, model_args_for_hardware_param)
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print

    px = 2
    pqkv = 32
    p_proj = px
    p_w1 = 64
    p_w2 = px

    pqkv = pqkv * num_heads

    from utils import manually_update_hardware_parallelism_param

    manually_update_hardware_parallelism_param(
        mg,
        pass_args={
            "fork2": {"din": [1, px], "dout": ([1, px], [1, px])},
            "blocks_0_norm1": {"din": [1, px], "dout": [1, px]},
            "blocks_0_attn": {"din": [1, px], "dattn": [1, pqkv], "dout": [1, p_proj]},
            "fifo": {"din": [1, px], "dout": [1, px]},
            "add": {"din": ([1, px], [1, px]), "dout": [1, px]},
            "fork2_1": {"din": [1, px], "dout": ([1, px], [1, px])},
            "blocks_0_norm2": {"din": [1, px], "dout": [1, px]},
            "blocks_0_mlp_fc1": {"din": [1, px], "dout": [1, p_w1]},
            "blocks_0_mlp_act": {"din": [1, p_w1], "dout": [1, p_w1]},
            "blocks_0_mlp_fc2": {"din": [1, p_w1], "dout": [1, px]},
            "fifo_1": {"din": [1, px], "dout": [1, px]},
            "add_1": {"din": ([1, px], [1, px]), "dout": [1, px]},
        },
    )
    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    # mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(
        mg, pass_args={"wait_time": 100, "wait_units": "ms", "batch_size": batch_size}
    )
    mg, _ = passes.emit_vivado_project_transform_pass(mg)

    simulate(
        skip_build=False, skip_test=False, simulator="questa", waves=True, gui=False
    )


if __name__ == "__main__":
    test_emit_verilog_vit()
