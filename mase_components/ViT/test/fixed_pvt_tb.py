#!/usr/bin/env python3

# This script tests the fixed point linear
import random, os, math, logging, sys
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append("/workspace/machop/")
sys.path.append("/workspace/components/testbench/ViT/")

from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from z_qlayers import quantize_to_int as q2i
from chop.models.manual.quant_utils import get_quantized_cls
from pvt_quant import QuantizedPyramidVisionTransformer
from z_qlayers import linear_data_pack
from ha_softmax import generate_table_hardware, generate_table_div_hardware

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        # width config
        self.samples = samples
        self.width_generate()
        # parameters config
        self.in_c = 3
        self.in_y = 224
        self.in_x = 224
        self.embed_dim = 384
        self.patch_size = 16
        self.num_patch = self.in_y * self.in_x // (self.patch_size**2)

        self.num_heads = 6
        self.mlp_ratio = 2

        self.pe_unroll_kernel_out = 24
        self.pe_unroll_in_c = 3
        self.pe_unroll_embed_dim = 8
        self.blk_unroll_qkv_dim = 2
        self.blk_unroll_hidden_features = 4

        self.num_classes = 10
        self.head_unroll_out_x = 1

        self.pe_iter_weight = int(
            (self.patch_size**2)
            * self.in_c
            * self.embed_dim
            / self.pe_unroll_kernel_out
            / self.pe_unroll_embed_dim
        )
        self.source_generate()
        self.outputs = RandomSink(
            samples=samples * int(self.num_classes / self.head_unroll_out_x),
            debug=debug,
        )
        self.samples = samples
        self.ref = self.sw_compute()

    def width_generate(self):
        din, din_f = 8, 3

        aff_msa_w, aff_msa_w_f = din, din_f
        aff_msa_b, aff_msa_b_f = 8, 3
        msa_din, msa_din_f = 8, 3
        wq, wq_f = 6, 4
        wkv, wkv_f = 6, 4
        wp, wp_f = 6, 4

        bq, bq_f = 6, 4
        bkv, bkv_f = 6, 4
        bp, bp_f = 6, 4

        dq, dq_f = 8, 3
        dk, dk_f = 8, 3
        dv, dv_f = 8, 3
        ds, ds_f = 8, 3
        softmax_exp, softmax_exp_f = 8, 5
        softmax_ds, softmax_ds_f = 8, 3
        div = 9
        dz, dz_f = 8, 3

        msa_o, msa_o_f = 8, 3

        aff_mlp_w, aff_mlp_w_f = msa_o + 1, msa_o_f
        aff_mlp_b, aff_mlp_b_f = 8, 3

        mlp_din, mlp_din_f = 8, 3
        fc1_w, fc1_w_f = 6, 4
        fc1_b, fc1_b_f = 6, 4
        mlp_hidden, mlp_hidden_f = 8, 3
        fc2_w, fc2_w_f = 6, 4
        fc2_b, fc2_b_f = 6, 4
        mlp_o, mlp_o_f = 8, 3
        self.w_config = {
            "patch_embed": {
                "patch_proj": {
                    "name": "integer",
                    "data_in_width": 8,
                    "data_in_frac_width": 5,
                    "weight_width": 8,
                    "weight_frac_width": 6,
                    "bias_width": 8,
                    "bias_frac_width": 5,
                },
            },
            "pos_add": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 5,
            },
            "block": {
                "affine_att": {
                    "mul": {
                        "name": "integer",
                        "data_in_width": aff_msa_w,
                        "data_in_frac_width": aff_msa_w_f,
                    },
                    "add": {
                        "name": "integer",
                        "data_in_width": aff_msa_b,
                        "data_in_frac_width": aff_msa_b_f,
                    },
                },
                "msa": {
                    "q_proj": {
                        "name": "integer",
                        "weight_width": wq,
                        "weight_frac_width": wq_f,
                        "data_in_width": msa_din,
                        "data_in_frac_width": msa_din_f,
                        "bias_width": bq,
                        "bias_frac_width": bq_f,
                    },
                    "kv_proj": {
                        "name": "integer",
                        "weight_width": wkv,
                        "weight_frac_width": wkv_f,
                        "data_in_width": msa_din,
                        "data_in_frac_width": msa_din_f,
                        "bias_width": bkv,
                        "bias_frac_width": bkv_f,
                    },
                    "z_proj": {
                        "name": "integer",
                        "weight_width": wp,
                        "weight_frac_width": wp_f,
                        "data_in_width": dz,
                        "data_in_frac_width": dz_f,
                        "bias_width": bp,
                        "bias_frac_width": bp_f,
                    },
                    "softmax": {
                        "name": "integer",
                        "exp_width": softmax_exp,
                        "exp_frac_width": softmax_exp_f,
                        "data_in_width": ds,
                        "data_in_frac_width": ds_f,
                        "data_out_width": softmax_ds,
                        "data_out_frac_width": softmax_ds_f,
                        "div_width": div,
                    },
                    "attn_matmul": {
                        "name": "integer",
                        "data_in_width": dq,
                        "data_in_frac_width": dq_f,
                        "weight_width": dk,
                        "weight_frac_width": dk_f,
                    },
                    "z_matmul": {
                        "name": "integer",
                        "data_in_width": softmax_ds,
                        "data_in_frac_width": softmax_ds_f,
                        "weight_width": dv,
                        "weight_frac_width": dv_f,
                    },
                },
                "add1": {
                    "name": "integer",
                    "data_in_width": msa_o,
                    "data_in_frac_width": msa_o_f,
                },
                "affine_mlp": {
                    "mul": {
                        "name": "integer",
                        "data_in_width": aff_mlp_w,
                        "data_in_frac_width": aff_mlp_w_f,
                    },
                    "add": {
                        "name": "integer",
                        "data_in_width": aff_mlp_b,
                        "data_in_frac_width": aff_mlp_b_f,
                    },
                },
                "mlp": {
                    "fc1_proj": {
                        "name": "integer",
                        "weight_width": fc1_w,
                        "weight_frac_width": fc1_w_f,
                        "data_in_width": mlp_din,
                        "data_in_frac_width": mlp_din_f,
                        "bias_width": fc1_b,
                        "bias_frac_width": fc1_b_f,
                    },
                    "mlp_relu": {
                        "name": "integer",
                        "bypass": True,
                        "data_in_width": mlp_hidden,
                        "data_in_frac_width": mlp_hidden_f,
                    },
                    "fc2_proj": {
                        "name": "integer",
                        "weight_width": fc2_w,
                        "weight_frac_width": fc2_w_f,
                        "data_in_width": mlp_hidden,
                        "data_in_frac_width": mlp_hidden_f,
                        "bias_width": fc2_b,
                        "bias_frac_width": fc2_b_f,
                    },
                },
                "add2": {
                    "name": "integer",
                    "data_in_width": mlp_o - 1,
                    "data_in_frac_width": mlp_o_f,
                },
            },
            # "pvt_norm":{
            #     "mul":{
            #         "name": "integer",
            #         "data_in_width":        aff_mlp_w,
            #         "data_in_frac_width":   aff_mlp_w_f,
            #         },
            #     "add":{
            #         "name": "integer",
            #         "data_in_width":        aff_mlp_b,
            #         "data_in_frac_width":   aff_mlp_b_f,
            #         },
            # },
            "head_proj": {
                "name": "integer",
                "data_in_width": mlp_o,
                "data_in_frac_width": mlp_o_f,
                "weight_width": 8,
                "weight_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
            },
        }
        self.ow, self.ow_f = 8, 3

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.w_config["patch_embed"]["patch_proj"]["data_in_width"],
            "IN_FRAC_WIDTH": self.w_config["patch_embed"]["patch_proj"][
                "data_in_frac_width"
            ],
            "PATCH_EMBED_W_WIDTH_3": self.w_config["patch_embed"]["patch_proj"][
                "weight_width"
            ],
            "PATCH_EMBED_W_FRAC_WIDTH_3": self.w_config["patch_embed"]["patch_proj"][
                "weight_frac_width"
            ],
            "PATCH_EMBED_B_WIDTH_3": self.w_config["patch_embed"]["patch_proj"][
                "bias_width"
            ],
            "PATCH_EMBED_B_FRAC_WIDTH_3": self.w_config["patch_embed"]["patch_proj"][
                "bias_frac_width"
            ],
            "POS_ADD_IN_WIDTH_3": self.w_config["pos_add"]["data_in_width"],
            "POS_ADD_IN_FRAC_WIDTH_3": self.w_config["pos_add"]["data_in_frac_width"],
            "BLOCK_IN_WIDTH": self.w_config["block"]["affine_att"]["mul"][
                "data_in_width"
            ],
            "BLOCK_IN_FRAC_WIDTH": self.w_config["block"]["affine_att"]["mul"][
                "data_in_frac_width"
            ],
            "BLOCK_AF_MSA_ADD_WIDTH": self.w_config["block"]["affine_att"]["add"][
                "data_in_width"
            ],
            "BLOCK_AF_MSA_ADD_FRAC_WIDTH": self.w_config["block"]["affine_att"]["add"][
                "data_in_frac_width"
            ],
            "BLOCK_MSA_IN_WIDTH": self.w_config["block"]["msa"]["q_proj"][
                "data_in_width"
            ],
            "BLOCK_MSA_IN_FRAC_WIDTH": self.w_config["block"]["msa"]["q_proj"][
                "data_in_frac_width"
            ],
            "BLOCK_WQ_WIDTH": self.w_config["block"]["msa"]["q_proj"]["weight_width"],
            "BLOCK_WQ_FRAC_WIDTH": self.w_config["block"]["msa"]["q_proj"][
                "weight_frac_width"
            ],
            "BLOCK_WK_WIDTH": self.w_config["block"]["msa"]["kv_proj"]["weight_width"],
            "BLOCK_WK_FRAC_WIDTH": self.w_config["block"]["msa"]["kv_proj"][
                "weight_frac_width"
            ],
            "BLOCK_WV_WIDTH": self.w_config["block"]["msa"]["kv_proj"]["weight_width"],
            "BLOCK_WV_FRAC_WIDTH": self.w_config["block"]["msa"]["kv_proj"][
                "weight_frac_width"
            ],
            "BLOCK_WP_WIDTH": self.w_config["block"]["msa"]["z_proj"]["weight_width"],
            "BLOCK_WP_FRAC_WIDTH": self.w_config["block"]["msa"]["z_proj"][
                "weight_frac_width"
            ],
            "BLOCK_BQ_WIDTH": self.w_config["block"]["msa"]["q_proj"]["bias_width"],
            "BLOCK_BQ_FRAC_WIDTH": self.w_config["block"]["msa"]["q_proj"][
                "bias_frac_width"
            ],
            "BLOCK_BK_WIDTH": self.w_config["block"]["msa"]["kv_proj"]["bias_width"],
            "BLOCK_BK_FRAC_WIDTH": self.w_config["block"]["msa"]["kv_proj"][
                "bias_frac_width"
            ],
            "BLOCK_BV_WIDTH": self.w_config["block"]["msa"]["kv_proj"]["bias_width"],
            "BLOCK_BV_FRAC_WIDTH": self.w_config["block"]["msa"]["kv_proj"][
                "bias_frac_width"
            ],
            "BLOCK_BP_WIDTH": self.w_config["block"]["msa"]["z_proj"]["bias_width"],
            "BLOCK_BP_FRAC_WIDTH": self.w_config["block"]["msa"]["z_proj"][
                "bias_frac_width"
            ],
            "BLOCK_DQ_WIDTH": self.w_config["block"]["msa"]["attn_matmul"][
                "data_in_width"
            ],
            "BLOCK_DQ_FRAC_WIDTH": self.w_config["block"]["msa"]["attn_matmul"][
                "data_in_frac_width"
            ],
            "BLOCK_DK_WIDTH": self.w_config["block"]["msa"]["attn_matmul"][
                "weight_width"
            ],
            "BLOCK_DK_FRAC_WIDTH": self.w_config["block"]["msa"]["attn_matmul"][
                "weight_frac_width"
            ],
            "BLOCK_DS_WIDTH": self.w_config["block"]["msa"]["z_matmul"][
                "data_in_width"
            ],
            "BLOCK_DS_FRAC_WIDTH": self.w_config["block"]["msa"]["z_matmul"][
                "data_in_frac_width"
            ],
            "BLOCK_DV_WIDTH": self.w_config["block"]["msa"]["z_matmul"]["weight_width"],
            "BLOCK_DV_FRAC_WIDTH": self.w_config["block"]["msa"]["z_matmul"][
                "weight_frac_width"
            ],
            "BLOCK_EXP_WIDTH": self.w_config["block"]["msa"]["softmax"]["exp_width"],
            "BLOCK_EXP_FRAC_WIDTH": self.w_config["block"]["msa"]["softmax"][
                "exp_frac_width"
            ],
            "BLOCK_DIV_WIDTH": self.w_config["block"]["msa"]["softmax"]["div_width"],
            "BLOCK_DS_SOFTMAX_WIDTH": self.w_config["block"]["msa"]["softmax"][
                "data_out_width"
            ],
            "BLOCK_DS_SOFTMAX_FRAC_WIDTH": self.w_config["block"]["msa"]["softmax"][
                "data_out_frac_width"
            ],
            "BLOCK_DZ_WIDTH": self.w_config["block"]["msa"]["z_proj"]["data_in_width"],
            "BLOCK_DZ_FRAC_WIDTH": self.w_config["block"]["msa"]["z_proj"][
                "data_in_frac_width"
            ],
            "BLOCK_AF_MLP_IN_WIDTH": self.w_config["block"]["affine_mlp"]["mul"][
                "data_in_width"
            ],
            "BLOCK_AF_MLP_IN_FRAC_WIDTH": self.w_config["block"]["affine_mlp"]["mul"][
                "data_in_frac_width"
            ],
            "BLOCK_AF_MLP_ADD_WIDTH": self.w_config["block"]["affine_mlp"]["add"][
                "data_in_width"
            ],
            "BLOCK_AF_MLP_ADD_FRAC_WIDTH": self.w_config["block"]["affine_mlp"]["add"][
                "data_in_frac_width"
            ],
            # mlp
            "BLOCK_MLP_IN_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "data_in_width"
            ],
            "BLOCK_MLP_IN_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "data_in_frac_width"
            ],
            "BLOCK_WEIGHT_I2H_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "weight_width"
            ],
            "BLOCK_WEIGHT_I2H_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "weight_frac_width"
            ],
            "BLOCK_BIAS_I2H_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "bias_width"
            ],
            "BLOCK_BIAS_I2H_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "bias_frac_width"
            ],
            "BLOCK_MLP_HAS_BIAS": 1,
            "BLOCK_MLP_HIDDEN_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "data_in_width"
            ],
            "BLOCK_MLP_HIDDEN_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "data_in_frac_width"
            ],
            "BLOCK_WEIGHT_H2O_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "weight_width"
            ],
            "BLOCK_WEIGHT_H2O_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "weight_frac_width"
            ],
            "BLOCK_BIAS_H2O_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "bias_width"
            ],
            "BLOCK_BIAS_H2O_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "bias_frac_width"
            ],
            "HEAD_IN_WIDTH": self.w_config["head_proj"]["data_in_width"],
            "HEAD_IN_FRAC_WIDTH": self.w_config["head_proj"]["data_in_frac_width"],
            "HEAD_W_WIDTH": self.w_config["head_proj"]["weight_width"],
            "HEAD_W_FRAC_WIDTH": self.w_config["head_proj"]["weight_frac_width"],
            "HEAD_B_WIDTH": self.w_config["head_proj"]["bias_width"],
            "HEAD_B_FRAC_WIDTH": self.w_config["head_proj"]["bias_frac_width"],
            "OUT_WIDTH": self.ow,
            "OUT_FRAC_WIDTH": self.ow_f,
            "PATCH_EMBED_IN_C_3": self.in_c,
            "PATCH_EMBED_IN_Y_3": self.in_y,
            "PATCH_EMEBD_IN_X_3": self.in_x,
            "PATCH_SIZE_3": self.patch_size,
            "PATCH_EMBED_EMBED_DIM_3": self.embed_dim,
            "PATCH_EMEBD_NUM_PATCH_3": self.num_patch,
            "PATCH_EMEBD_UNROLL_KERNEL_OUT_3": self.pe_unroll_kernel_out,
            "PATCH_EMEBD_UNROLL_IN_C_3": self.pe_unroll_in_c,
            "PATCH_EMBED_UNROLL_EMBED_DIM_3": self.pe_unroll_embed_dim,
            "NUM_HEADS": self.num_heads,
            "MLP_RATIO": self.mlp_ratio,
            "BLOCK_UNROLL_WQKV_DIM": self.blk_unroll_qkv_dim,
            "BLOCK_UNROLL_HIDDEN_FEATURES": self.blk_unroll_hidden_features,
            "NUM_CLASSES": self.num_classes,
            "HEAD_UNROLL_OUT_X": self.head_unroll_out_x,
        }

    def source_generate(self):
        samples = self.samples
        torch.manual_seed(2)
        self.x = torch.randn((samples, self.in_c, self.in_y, self.in_x))
        input_tensor = q2i(
            self.x,
            self.w_config["patch_embed"]["patch_proj"]["data_in_width"],
            self.w_config["patch_embed"]["patch_proj"]["data_in_frac_width"],
        )
        x_in = input_tensor.permute(0, 2, 3, 1).reshape(-1, self.pe_unroll_in_c)

        x_in = x_in.flip(0).tolist()
        self.inputs = RandomSource(
            max_stalls=0,
            name="data_in",
            samples=samples
            * int(self.in_x * self.in_y * self.in_c / self.pe_unroll_in_c),
            num=self.pe_unroll_in_c,
            data_specify=x_in,
            debug=debug,
        )
        self.pvt = QuantizedPyramidVisionTransformer(
            img_size=self.in_y,
            in_chans=self.in_c,
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            embed_dims=[self.embed_dim, self.embed_dim, self.embed_dim, self.embed_dim],
            num_heads=[self.num_heads, self.num_heads, self.num_heads, self.num_heads],
            mlp_ratios=[self.mlp_ratio, self.mlp_ratio, self.mlp_ratio, self.mlp_ratio],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            depths=[1, 1, 1, 1],
            num_stages=1,
            config=self.w_config,
        )
        self.patch_source_generate(self.pvt)
        self.block_source_generate(self.pvt.block1[0])

    def patch_source_generate(self, pvt):
        samples = self.samples
        patch_w_1 = q2i(
            pvt.patch_embed1.proj.weight,
            self.w_config["patch_embed"]["patch_proj"]["weight_width"],
            self.w_config["patch_embed"]["patch_proj"]["weight_frac_width"],
        )
        patch_b_1 = q2i(
            pvt.patch_embed1.proj.bias,
            self.w_config["patch_embed"]["patch_proj"]["bias_width"],
            self.w_config["patch_embed"]["patch_proj"]["bias_frac_width"],
        )
        cls_in = q2i(
            self.pvt.cls_token,
            self.w_config["pos_add"]["data_in_width"],
            self.w_config["pos_add"]["data_in_frac_width"],
        )
        # NOTE: only 1 layer patch
        # position embed
        patch_embed_H = self.in_y // self.patch_size
        patch_embed_W = patch_embed_H
        H = patch_embed_H
        W = patch_embed_W
        pos_embed = self.pvt.pos_embed1
        pos_embed_ = (
            F.interpolate(
                pos_embed[:, 1:]
                .reshape(1, patch_embed_H, patch_embed_W, -1)
                .permute(0, 3, 1, 2),
                size=(H, W),
                mode="bilinear",
            )
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
        pos_in = q2i(
            pos_embed,
            self.w_config["pos_add"]["data_in_width"],
            self.w_config["pos_add"]["data_in_frac_width"],
        )
        head_w = q2i(
            self.pvt.head.weight,
            self.w_config["head_proj"]["weight_width"],
            self.w_config["head_proj"]["weight_frac_width"],
        )
        head_b = q2i(
            self.pvt.head.bias,
            self.w_config["head_proj"]["bias_width"],
            self.w_config["head_proj"]["bias_frac_width"],
        )
        # parameters packs
        pe_w_in, pe_b_in = self.conv_pack(
            weight=patch_w_1,
            bias=patch_b_1,
            in_channels=self.in_c,
            kernel_size=[self.patch_size, self.patch_size],
            out_channels=self.embed_dim,
            unroll_in_channels=self.pe_unroll_in_c,
            unroll_kernel_out=self.pe_unroll_kernel_out,
            unroll_out_channels=self.pe_unroll_embed_dim,
        )

        cls_in = linear_data_pack(
            samples,
            cls_in.repeat(self.samples, 1, 1),
            in_y=1,
            unroll_in_y=1,
            in_x=self.embed_dim,
            unroll_in_x=self.pe_unroll_embed_dim,
        )
        pos_in = linear_data_pack(
            samples,
            pos_in.repeat(self.samples, 1, 1),
            in_y=self.num_patch + 1,
            unroll_in_y=1,
            in_x=self.embed_dim,
            unroll_in_x=self.pe_unroll_embed_dim,
        )
        h_w_in = linear_data_pack(
            samples,
            head_w.repeat(self.samples, 1, 1),
            in_y=self.num_classes,
            unroll_in_y=self.head_unroll_out_x,
            in_x=self.embed_dim,
            unroll_in_x=self.pe_unroll_embed_dim,
        )

        h_b_in = linear_data_pack(
            samples,
            head_b.repeat(self.samples, 1, 1),
            in_y=self.num_classes,
            unroll_in_y=self.head_unroll_out_x,
            in_x=1,
            unroll_in_x=1,
        )

        cls_in.reverse()
        pos_in.reverse()
        h_w_in.reverse()
        h_b_in.reverse()

        self.patch_embed_bias = RandomSource(
            max_stalls=0,
            name="patch_embed_bias",
            samples=samples * int(self.embed_dim / self.pe_unroll_embed_dim),
            num=self.pe_unroll_embed_dim,
            data_specify=pe_b_in,
            debug=debug,
        )
        self.patch_embed_weight = RandomSource(
            max_stalls=0,
            name="patch_embed_weight",
            samples=samples * self.pe_iter_weight,
            num=self.pe_unroll_kernel_out * self.pe_unroll_embed_dim,
            data_specify=pe_w_in,
            debug=debug,
        )
        self.cls_token = RandomSource(
            max_stalls=0,
            name="cls_token_data",
            samples=samples * (self.embed_dim // self.pe_unroll_embed_dim),
            num=self.pe_unroll_embed_dim,
            data_specify=cls_in,
            debug=debug,
        )
        self.pos_embed = RandomSource(
            max_stalls=0,
            name="pos_embed_data",
            samples=samples
            * (self.num_patch + 1)
            * (self.embed_dim // self.pe_unroll_embed_dim),
            num=self.pe_unroll_embed_dim,
            data_specify=pos_in,
            debug=debug,
        )
        self.head_bias = RandomSource(
            max_stalls=0,
            name="head_bias",
            samples=samples * int(self.num_classes / self.head_unroll_out_x),
            num=self.head_unroll_out_x,
            data_specify=h_b_in,
            debug=debug,
        )
        self.head_weight = RandomSource(
            max_stalls=0,
            name="head_weight",
            samples=samples
            * int(self.num_classes / self.head_unroll_out_x)
            * int(self.embed_dim / self.pe_unroll_embed_dim),
            num=self.head_unroll_out_x * self.pe_unroll_embed_dim,
            data_specify=h_w_in,
            debug=debug,
        )

    def data_pack(self, in_temp, np, d, p, s):
        # assum in_temp.shape = (samples, batch = 1, N,dim)
        in_temp = in_temp.to(torch.int).reshape(self.samples, np * p, d * s)
        ref = []
        for i in range(self.samples):
            re_tensor = rearrange(
                in_temp[i], "(np p) (d s) -> np (p d) s", np=np, d=d, p=p, s=s
            )
            ex_tensor = torch.zeros(np, d * p, s, dtype=int)
            for b in range(np):
                for i in range(d):
                    for j in range(p):
                        ex_tensor[b][i * p + j] = re_tensor[b][j * d + i]
            output_tensor = rearrange(
                ex_tensor, "np (d p) s -> (np d) (p s)", np=np, d=d, p=p, s=s
            )
            output = output_tensor.tolist()
            ref = ref + output
        return ref

    def block_source_generate(self, qblock):
        samples = self.samples
        w_config = self.w_config["block"]
        att = qblock.attn
        aff_att = qblock.norm1
        in_y = self.num_patch + 1
        in_x = self.embed_dim
        qkv_x = in_x
        out_x = in_x
        unroll_in_y = 1
        unroll_in_x = self.pe_unroll_embed_dim
        unroll_qkv_x = self.blk_unroll_qkv_dim
        unroll_out_x = unroll_in_x
        self.msa_data_generate(
            att,
            in_x,
            unroll_in_x,
            qkv_x,
            unroll_qkv_x,
            self.num_heads,
            out_x,
            unroll_out_x,
        )
        mlp = qblock.mlp
        in_features = in_x
        unroll_in_features = unroll_in_x
        hidden_features = self.mlp_ratio * in_features
        unroll_hidden_features = self.blk_unroll_hidden_features
        out_features = in_x
        unroll_out_features = unroll_in_x
        self.mlp_data_generate(
            mlp,
            in_features,
            hidden_features,
            out_features,
            unroll_in_features,
            unroll_hidden_features,
            unroll_out_features,
        )

        num = in_y * in_x // (unroll_in_x * unroll_in_y)
        in_size = unroll_in_x * unroll_in_y
        aff_att_w, aff_att_b = self.aff_data_generate(
            w_config["affine_att"], aff_att, num, in_size
        )
        aff_mlp = qblock.norm2
        aff_mlp_w, aff_mlp_b = self.aff_data_generate(
            w_config["affine_mlp"], aff_mlp, num, in_size
        )
        self.aff_att_weight = RandomSource(
            max_stalls=0,
            samples=samples * num,
            num=in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=aff_att_w,
        )
        self.aff_att_bias = RandomSource(
            max_stalls=0,
            samples=samples * num,
            num=in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=aff_att_b,
        )

        self.aff_mlp_weight = RandomSource(
            max_stalls=0,
            samples=samples * num,
            num=in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=aff_mlp_w,
        )
        self.aff_mlp_bias = RandomSource(
            max_stalls=0,
            samples=samples * num,
            num=in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=aff_mlp_b,
        )

    def aff_data_generate(self, config, qaff, num, in_size):
        fixed_aff = qaff
        w = fixed_aff.weight
        b = fixed_aff.bias
        weight_in = (
            q2i(w, config["mul"]["data_in_width"], config["mul"]["data_in_frac_width"])
            .repeat(self.samples * num, in_size)
            .tolist()
        )

        bias_in = (
            q2i(b, config["add"]["data_in_width"], config["add"]["data_in_frac_width"])
            .repeat(self.samples * num, in_size)
            .tolist()
        )
        weight_in.reverse()
        bias_in.reverse()
        return weight_in, bias_in

    def msa_data_generate(
        self,
        qatt,
        in_x,
        unroll_in_x,
        qkv_x,
        unroll_qkv_x,
        num_heads,
        out_x,
        unroll_out_x,
    ):
        # generate data
        samples = self.samples
        in_depth = in_x // unroll_in_x
        in_size = unroll_in_x
        wqkv_parallelism = unroll_qkv_x
        wqkv_num_parallelism = qkv_x // (unroll_qkv_x * num_heads)
        wp_parallelism = unroll_out_x
        wp_num_parallelism = out_x // unroll_out_x
        wp_depth = qkv_x // (unroll_qkv_x * num_heads)
        wp_size = num_heads * unroll_qkv_x
        dim = in_size * in_depth
        config = self.w_config["block"]["msa"]
        att = qatt
        att_wq = q2i(
            att.q.weight,
            config["q_proj"]["weight_width"],
            config["q_proj"]["weight_frac_width"],
        )
        att_wkv = q2i(
            att.kv.weight,
            config["kv_proj"]["weight_width"],
            config["kv_proj"]["weight_frac_width"],
        )
        wqkv_tensor = torch.cat((att_wq, att_wkv), 0)
        wqkv_tensor = wqkv_tensor.reshape(3, in_x, in_x)
        wqkv_tensor = wqkv_tensor.reshape(in_x * 3, in_x).repeat(samples, 1, 1)

        att_bq = q2i(
            att.q.bias,
            config["q_proj"]["bias_width"],
            config["q_proj"]["bias_frac_width"],
        )
        att_bkv = q2i(
            att.kv.bias,
            config["kv_proj"]["bias_width"],
            config["kv_proj"]["bias_frac_width"],
        )
        bqkv_tensor = torch.cat((att_bq, att_bkv), 0)
        bqkv_tensor = bqkv_tensor.reshape(3, in_x)
        bqkv_tensor = bqkv_tensor.reshape(-1).repeat(samples, 1)

        wp_tensor = q2i(
            att.proj.weight,
            config["z_proj"]["weight_width"],
            config["z_proj"]["weight_frac_width"],
        ).repeat(samples, 1, 1)
        bp_tensor = q2i(
            att.proj.bias,
            config["z_proj"]["bias_width"],
            config["z_proj"]["bias_frac_width"],
        ).repeat(samples, 1)

        logger.debug(
            "input data: \n\
        wqkv_tensor = \n{}\n\
        bqkv_tensor = \n{}\n\
        wp_tensor = \n{}\n\
        bp_tensor = \n{}\n\
        ".format(
                wqkv_tensor, bqkv_tensor, wp_tensor, bp_tensor
            )
        )
        # generate hash table
        exp_table = generate_table_hardware(
            att.scale,
            config["softmax"]["data_in_width"],
            config["softmax"]["data_in_frac_width"],
            config["softmax"]["exp_width"],
            config["softmax"]["exp_frac_width"],
        ).tolist()
        div_table = generate_table_div_hardware(
            config["softmax"]["div_width"],
            config["softmax"]["data_out_width"],
            config["softmax"]["data_out_frac_width"],
        ).tolist()
        with open(r"exp_init.mem", "w") as fp:
            for item in exp_table:
                # write each item on a new lineformat(addr[i] ,f'0{width}b'
                fp.write(
                    "%s\n" % format(item, f'0{config["softmax"]["exp_width"]//4}x')
                )
        with open(r"div_init.mem", "w") as fp:
            for item in div_table:
                # write each item on a new line
                fp.write(
                    "%s\n" % format(item, f'0{config["softmax"]["data_out_width"]//4}x')
                )
        # data_pack
        wqkv = wqkv_tensor.reshape(
            samples, 3, num_heads, wqkv_num_parallelism, wqkv_parallelism, dim
        ).permute(1, 0, 3, 2, 4, 5)
        wqkv = wqkv.reshape(3, samples, dim, dim)
        bqkv = bqkv_tensor.reshape(
            samples, 3, num_heads, wqkv_num_parallelism, wqkv_parallelism
        ).permute(1, 0, 3, 2, 4)
        bqkv = bqkv.reshape(3, samples, dim)

        wp = wp_tensor.reshape(
            samples * dim, num_heads, wqkv_num_parallelism, wqkv_parallelism
        )
        wp = wp.permute(0, 2, 1, 3).reshape(samples, dim, dim)

        wq = wqkv[0]
        wk = wqkv[1]
        wv = wqkv[2]

        bq = bqkv[0]
        bk = bqkv[1]
        bv = bqkv[2]
        wq_in = self.data_pack(
            wq, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        wk_in = self.data_pack(
            wk, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        wv_in = self.data_pack(
            wv, wqkv_num_parallelism, in_depth, num_heads * wqkv_parallelism, in_size
        )
        wp_in = self.data_pack(
            wp,
            wp_num_parallelism,
            wqkv_num_parallelism,
            wp_parallelism,
            num_heads * wqkv_parallelism,
        )

        bq_in = self.data_pack(
            bq, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        bk_in = self.data_pack(
            bk, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        bv_in = self.data_pack(
            bv, 1, wqkv_num_parallelism, 1, num_heads * wqkv_parallelism
        )
        bp_in = self.data_pack(bp_tensor, 1, wp_num_parallelism, 1, wp_parallelism)

        wq_in.reverse()
        wk_in.reverse()
        wv_in.reverse()
        wp_in.reverse()
        bq_in.reverse()
        bk_in.reverse()
        bv_in.reverse()
        bp_in.reverse()

        self.weight_q = RandomSource(
            max_stalls=0,
            name="weight_q",
            samples=samples * in_depth * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism * in_size,
            data_specify=wq_in,
            debug=debug,
        )
        self.weight_k = RandomSource(
            max_stalls=0,
            name="weight_k",
            samples=samples * in_depth * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism * in_size,
            data_specify=wk_in,
            debug=debug,
        )
        self.weight_v = RandomSource(
            max_stalls=0,
            name="weight_v",
            samples=samples * in_depth * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism * in_size,
            data_specify=wv_in,
            debug=debug,
        )
        self.weight_p = RandomSource(
            max_stalls=0,
            name="weight_p",
            samples=samples * wp_depth * wp_num_parallelism,
            num=wp_parallelism * wp_size,
            data_specify=wp_in,
            debug=debug,
        )
        self.bias_q = RandomSource(
            max_stalls=0,
            name="bias_q",
            samples=samples * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism,
            data_specify=bq_in,
            debug=debug,
        )
        self.bias_k = RandomSource(
            max_stalls=0,
            name="bias_k",
            samples=samples * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism,
            data_specify=bk_in,
            debug=debug,
        )
        self.bias_v = RandomSource(
            max_stalls=0,
            name="bias_v",
            samples=samples * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism,
            data_specify=bv_in,
            debug=debug,
        )
        self.bias_p = RandomSource(
            max_stalls=0,
            name="bias_p",
            samples=samples * wp_num_parallelism,
            num=wp_parallelism,
            data_specify=bp_in,
            debug=debug,
        )

    def mlp_data_generate(
        self,
        qmlp,
        in_features,
        hidden_features,
        out_features,
        unroll_in_features,
        unroll_hidden_features,
        unroll_out_features,
    ):
        samples = self.samples
        depth_in_features = in_features // unroll_in_features
        depth_hidden_features = hidden_features // unroll_hidden_features
        depth_out_features = out_features // unroll_out_features
        w_config = self.w_config["block"]["mlp"]
        mlp = qmlp
        weight1_tensor = q2i(
            mlp.fc1.weight,
            w_config["fc1_proj"]["weight_width"],
            w_config["fc1_proj"]["weight_frac_width"],
        )

        bias1_tensor = q2i(
            mlp.fc1.bias,
            w_config["fc1_proj"]["bias_width"],
            w_config["fc1_proj"]["bias_frac_width"],
        )

        weight2_tensor = q2i(
            mlp.fc2.weight,
            w_config["fc2_proj"]["weight_width"],
            w_config["fc2_proj"]["weight_frac_width"],
        )

        bias2_tensor = q2i(
            mlp.fc2.bias,
            w_config["fc2_proj"]["bias_width"],
            w_config["fc2_proj"]["bias_frac_width"],
        )
        weight1_in = linear_data_pack(
            samples,
            weight1_tensor.repeat(samples, 1, 1),
            in_y=hidden_features,
            in_x=in_features,
            unroll_in_y=unroll_hidden_features,
            unroll_in_x=unroll_in_features,
        )
        bias1_in = linear_data_pack(
            samples,
            bias1_tensor.repeat(samples, 1, 1),
            in_y=hidden_features,
            in_x=1,
            unroll_in_y=unroll_hidden_features,
            unroll_in_x=1,
        )
        weight2_in = linear_data_pack(
            samples,
            weight2_tensor.repeat(samples, 1, 1),
            in_y=out_features,
            in_x=hidden_features,
            unroll_in_y=unroll_out_features,
            unroll_in_x=unroll_hidden_features,
        )
        bias2_in = linear_data_pack(
            samples,
            bias2_tensor.repeat(samples, 1, 1),
            in_y=out_features,
            in_x=1,
            unroll_in_y=unroll_out_features,
            unroll_in_x=1,
        )
        weight1_in.reverse()
        bias1_in.reverse()
        weight2_in.reverse()
        bias2_in.reverse()
        self.bias1 = RandomSource(
            max_stalls=0,
            name="bias1",
            samples=samples * depth_hidden_features,
            num=unroll_hidden_features,
            data_specify=bias1_in,
            debug=debug,
        )
        self.bias2 = RandomSource(
            max_stalls=0,
            name="bias2",
            samples=samples * depth_out_features,
            num=unroll_out_features,
            data_specify=bias2_in,
            debug=debug,
        )
        self.weight1 = RandomSource(
            max_stalls=0,
            name="weight1",
            samples=samples * depth_hidden_features * depth_in_features,
            num=unroll_hidden_features * unroll_in_features,
            data_specify=weight1_in,
            debug=debug,
        )
        self.weight2 = RandomSource(
            max_stalls=0,
            name="weight2",
            samples=samples * depth_out_features * depth_hidden_features,
            num=unroll_out_features * unroll_hidden_features,
            data_specify=weight2_in,
            debug=debug,
        )

    def sw_compute(self):
        data_out = self.pvt(self.x)
        output = linear_data_pack(
            self.samples,
            q2i(data_out, self.ow, self.ow_f),
            in_y=1,
            in_x=self.num_classes,
            unroll_in_y=1,
            unroll_in_x=self.head_unroll_out_x,
        )
        return output

    def conv_pack(
        self,
        weight,
        bias,
        in_channels,
        kernel_size,
        out_channels,
        unroll_in_channels,
        unroll_kernel_out,
        unroll_out_channels,
    ):
        samples = self.samples
        # requires input as a quantized int format
        # weight_pack
        # from (oc,ic/u_ic,u_ic,h,w) to (ic/u_ic,h*w,u_ic,oc)
        reorder_w_tensor = (
            weight.repeat(samples, 1, 1, 1, 1)
            .reshape(
                samples,
                out_channels,
                int(in_channels / unroll_in_channels),
                unroll_in_channels,
                kernel_size[0] * kernel_size[1],
            )
            .permute(0, 2, 4, 3, 1)
        )

        # reverse the final 2 dimension
        # from(samples, int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_kernel_out, int(out_channels/unroll_out_channels), unroll_out_channels)
        # to  (samples, int(out_channels/unroll_out_channels), int(kernel_height * kernel_width * in_channels / unroll_kernel_out), unroll_out_channels, unroll_kernel_out)
        w_tensor = reorder_w_tensor.reshape(
            samples,
            int(kernel_size[0] * kernel_size[1] * in_channels / unroll_kernel_out),
            unroll_kernel_out,
            int(out_channels / unroll_out_channels),
            unroll_out_channels,
        ).permute(0, 3, 1, 4, 2)

        w_tensor = w_tensor.reshape(
            -1,
            unroll_out_channels * unroll_kernel_out,
        )
        w_in = w_tensor.type(torch.int).flip(0).tolist()
        # bias_pack
        bias_tensor = bias.repeat(samples, 1).reshape(-1, unroll_out_channels)
        b_in = bias_tensor.type(torch.int).flip(0).tolist()
        return w_in, b_in


@cocotb.test()
async def test_fixed_linear(dut):
    # TODO:
    """Test integer based vector mult"""
    samples = 1
    test_case = VerificationCase(samples=samples)

    # Reset cycle
    await Timer(20, units="ns")
    dut.rst.value = 1
    await Timer(100, units="ns")
    dut.rst.value = 0

    # Create a 10ns-period clock on port clk
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    await Timer(500, units="ns")

    # Synchronize with the clock
    dut.patch_embed_weight_3_valid.value = 0
    dut.patch_embed_bias_3_valid.value = 0
    dut.cls_token_valid.value = 0
    dut.pos_embed_in_valid.value = 0

    dut.af_msa_weight_valid.value = 0
    dut.af_msa_bias_valid.value = 0
    dut.weight_q_valid.value = 0
    dut.weight_k_valid.value = 0
    dut.weight_v_valid.value = 0
    dut.weight_p_valid.value = 0
    dut.bias_q_valid.value = 0
    dut.bias_k_valid.value = 0
    dut.bias_v_valid.value = 0
    dut.bias_p_valid.value = 0

    dut.af_mlp_weight_valid.value = 0
    dut.af_mlp_bias_valid.value = 0
    dut.weight_in2hidden_valid.value = 0
    dut.weight_hidden2out_valid.value = 0
    dut.bias_in2hidden_valid.value = 0
    dut.bias_hidden2out_valid.value = 0

    dut.head_weight_valid.value = 0
    dut.head_bias_valid.value = 0

    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    # debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    # debug_state(dut, "Post-clk")
    # debug_state(dut, "Pre-clk")
    await FallingEdge(dut.clk)
    # debug_state(dut, "Post-clk")

    done = False
    cdin = 0
    cpatch_out = 0
    cpose_out = 0
    cmsa_out = 0
    cblock_out = 0
    chead_in = 0
    cafmsa_out = 0
    cmsa_sa_out = 0
    for i in range(samples * 10000000):
        await FallingEdge(dut.clk)
        # debug_state(dut, "Post-clk")
        dut.rst.value = 0

        dut.patch_embed_bias_3_valid.value = test_case.patch_embed_bias.pre_compute()
        dut.patch_embed_weight_3_valid.value = (
            test_case.patch_embed_weight.pre_compute()
        )
        dut.cls_token_valid.value = test_case.cls_token.pre_compute()
        dut.pos_embed_in_valid.value = test_case.pos_embed.pre_compute()

        dut.af_msa_weight_valid.value = test_case.aff_att_weight.pre_compute()
        dut.af_msa_bias_valid.value = test_case.aff_att_bias.pre_compute()
        dut.weight_q_valid.value = test_case.weight_q.pre_compute()
        dut.weight_k_valid.value = test_case.weight_k.pre_compute()
        dut.weight_v_valid.value = test_case.weight_v.pre_compute()
        dut.weight_p_valid.value = test_case.weight_p.pre_compute()
        dut.bias_q_valid.value = test_case.bias_q.pre_compute()
        dut.bias_k_valid.value = test_case.bias_k.pre_compute()
        dut.bias_v_valid.value = test_case.bias_v.pre_compute()
        dut.bias_p_valid.value = test_case.bias_p.pre_compute()

        dut.af_mlp_weight_valid.value = test_case.aff_mlp_weight.pre_compute()
        dut.af_mlp_bias_valid.value = test_case.aff_mlp_bias.pre_compute()
        dut.weight_in2hidden_valid.value = test_case.weight1.pre_compute()
        dut.bias_in2hidden_valid.value = test_case.bias1.pre_compute()
        dut.weight_hidden2out_valid.value = test_case.weight2.pre_compute()
        dut.bias_hidden2out_valid.value = test_case.bias2.pre_compute()

        dut.head_weight_valid.value = test_case.head_weight.pre_compute()
        dut.head_bias_valid.value = test_case.head_bias.pre_compute()

        dut.data_in_valid.value = test_case.inputs.pre_compute()

        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(dut.data_out_valid)
        await Timer(1, units="ns")
        # start input data
        (
            dut.patch_embed_weight_3_valid.value,
            dut.patch_embed_weight_3.value,
        ) = test_case.patch_embed_weight.compute(dut.patch_embed_weight_3_ready.value)
        (
            dut.patch_embed_bias_3_valid.value,
            dut.patch_embed_bias_3.value,
        ) = test_case.patch_embed_bias.compute(dut.patch_embed_bias_3_ready.value)
        dut.cls_token_valid.value, dut.cls_token.value = test_case.cls_token.compute(
            dut.cls_token_ready.value
        )
        (
            dut.pos_embed_in_valid.value,
            dut.pos_embed_in.value,
        ) = test_case.pos_embed.compute(dut.pos_embed_in_ready.value)

        (
            dut.af_msa_weight_valid.value,
            dut.af_msa_weight.value,
        ) = test_case.aff_att_weight.compute(dut.af_msa_weight_ready.value)
        (
            dut.af_msa_bias_valid.value,
            dut.af_msa_bias.value,
        ) = test_case.aff_att_bias.compute(dut.af_msa_bias_ready.value)
        dut.weight_q_valid.value, dut.weight_q.value = test_case.weight_q.compute(
            dut.weight_q_ready.value
        )
        dut.weight_k_valid.value, dut.weight_k.value = test_case.weight_k.compute(
            dut.weight_k_ready.value
        )
        dut.weight_v_valid.value, dut.weight_v.value = test_case.weight_v.compute(
            dut.weight_v_ready.value
        )
        dut.weight_p_valid.value, dut.weight_p.value = test_case.weight_p.compute(
            dut.weight_p_ready.value
        )

        dut.bias_q_valid.value, dut.bias_q.value = test_case.bias_q.compute(
            dut.bias_q_ready.value
        )
        dut.bias_k_valid.value, dut.bias_k.value = test_case.bias_k.compute(
            dut.bias_k_ready.value
        )
        dut.bias_v_valid.value, dut.bias_v.value = test_case.bias_v.compute(
            dut.bias_v_ready.value
        )
        dut.bias_p_valid.value, dut.bias_p.value = test_case.bias_p.compute(
            dut.bias_p_ready.value
        )

        (
            dut.af_mlp_weight_valid.value,
            dut.af_mlp_weight.value,
        ) = test_case.aff_mlp_weight.compute(dut.af_mlp_weight_ready.value)
        (
            dut.af_mlp_bias_valid.value,
            dut.af_mlp_bias.value,
        ) = test_case.aff_mlp_bias.compute(dut.af_mlp_bias_ready.value)
        (
            dut.weight_in2hidden_valid.value,
            dut.weight_in2hidden.value,
        ) = test_case.weight1.compute(dut.weight_in2hidden_ready.value)
        (
            dut.weight_hidden2out_valid.value,
            dut.weight_hidden2out.value,
        ) = test_case.weight2.compute(dut.weight_hidden2out_ready.value)
        (
            dut.bias_in2hidden_valid.value,
            dut.bias_in2hidden.value,
        ) = test_case.bias1.compute(dut.bias_in2hidden_ready.value)
        (
            dut.bias_hidden2out_valid.value,
            dut.bias_hidden2out.value,
        ) = test_case.bias2.compute(dut.bias_hidden2out_ready.value)

        (
            dut.head_weight_valid.value,
            dut.head_weight.value,
        ) = test_case.head_weight.compute(dut.head_weight_ready.value)
        dut.head_bias_valid.value, dut.head_bias.value = test_case.head_bias.compute(
            dut.head_bias_ready.value
        )
        dut.data_in_valid.value, dut.data_in.value = test_case.inputs.compute(
            dut.data_in_ready.value
        )

        await Timer(1, units="ns")

        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        # wave_check(dut)
        if dut.data_in_valid.value == 1 and dut.data_in_ready.value == 1:
            cdin += 1
        if (
            dut.patch_embed_out_3_valid.value == 1
            and dut.patch_embed_out_3_ready.value == 1
        ):
            cpatch_out += 1
        if dut.pos_embed_out_valid.value == 1 and dut.pos_embed_out_ready.value == 1:
            cpose_out += 1
        if (
            dut.block_inst.af_msa_out_valid.value == 1
            and dut.block_inst.af_msa_out_ready.value == 1
        ):
            cafmsa_out += 1
        if (
            dut.block_inst.msa_inst.sa_out_valid.value == 1
            and dut.block_inst.msa_inst.sa_out_ready.value == 1
        ):
            cmsa_sa_out += 1
        if (
            dut.block_inst.msa_out_valid.value == 1
            and dut.block_inst.msa_out_ready.value == 1
        ):
            cmsa_out += 1
        if dut.block_out_valid.value == 1 and dut.block_out_ready.value == 1:
            cblock_out += 1
        if dut.head_in_valid.value == 1 and dut.head_in_ready.value == 1:
            chead_in += 1
        print("cdin = ", cdin)
        print("cpatch_out = ", cpatch_out)
        print("cpose_out = ", cpose_out)
        print(
            "{},{},cafmsa_out = {}".format(
                dut.block_inst.af_msa_out_valid.value,
                dut.block_inst.af_msa_out_ready.value,
                cafmsa_out,
            )
        )
        print("cmsa_sa_out = ", cmsa_sa_out)
        print("cmsa_out = ", cmsa_out)
        print("cblock_out = ", cblock_out)
        print("chead_in = ", chead_in)
        # if i % 1000 == 0:
        if (
            test_case.outputs.is_full()
            # and test_case.head_bias.is_empty()
            # and test_case.head_weight.is_empty()
            # and test_case.cls_token.is_empty()
            # and test_case.pos_embed.is_empty()
            # and test_case.patch_embed_bias.is_empty()
            # and test_case.patch_embed_weight.is_empty()
            # and test_case.weight1.is_empty()
            # and test_case.bias1.is_empty()
            # and test_case.weight2.is_empty()
            # and test_case.bias2.is_empty()
            # and test_case.weight_q.is_empty()
            # and test_case.weight_k.is_empty()
            # and test_case.weight_v.is_empty()
            # and test_case.weight_p.is_empty()
            # and test_case.bias_q.is_empty()
            # and test_case.bias_k.is_empty()
            # and test_case.bias_v.is_empty()
            # and test_case.bias_p.is_empty()
            # and test_case.inputs.is_empty()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def wave_check(dut):
    logger.debug(
        "wave_check:\n\
                {},{} pos_embed_out\n\
                {},{} block = {}\n\
                {},{} res_msa = {}\n\
                {},{} mlp_out = {}\n\
                {},{} head_in\n\
                {},{} data_out\n\
                ".format(
            dut.pos_embed_out_valid.value,
            dut.pos_embed_out_ready.value,
            dut.block_out_valid.value,
            dut.block_out_ready.value,
            [int(i) for i in dut.block_out.value],
            dut.block_inst.res_msa_valid.value,
            dut.block_inst.res_msa_ready.value,
            [int(i) for i in dut.block_inst.res_msa.value],
            dut.block_inst.mlp_out_valid.value,
            dut.block_inst.mlp_out_ready.value,
            [int(i) for i in dut.block_inst.mlp_out.value],
            dut.head_in_valid.value,
            dut.head_in_ready.value,
            dut.data_out_valid.value,
            dut.data_out_ready.value,
        )
    )


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/ViT/fixed_pvt.sv",
        "../../../../components/ViT/fixed_block.sv",
        "../../../../components/ViT/hash_softmax.sv",
        "../../../../components/ViT/affine_layernorm.sv",
        "../../../../components/ViT/fixed_mlp.sv",
        "../../../../components/ViT/fixed_msa.sv",
        "../../../../components/ViT/fixed_patch_embed.sv",
        "../../../../components/attention/fixed_self_att.sv",
        "../../../../components/attention/fixed_att.sv",
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/cast/fixed_rounding.sv",
        "../../../../components/activations/fixed_relu.sv",
        "../../../../components/linear/fixed_2d_linear.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/conv/convolution.sv",
        "../../../../components/conv/padding.sv",
        "../../../../components/conv/roller.sv",
        "../../../../components/conv/sliding_window.sv",
        "../../../../components/common/wrap_data.sv",
        "../../../../components/common/cut_data.sv",
        "../../../../components/common/fifo.sv",
        "../../../../components/common/unpacked_fifo.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/blk_mem_gen_0.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpacked_skid_buffer.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/common/split2.sv",
        "../../../../components/fixed_arith/fixed_matmul_core.sv",
        "../../../../components/fixed_arith/fixed_dot_product.sv",
        "../../../../components/fixed_arith/fixed_accumulator.sv",
        "../../../../components/fixed_arith/fixed_vector_mult.sv",
        "../../../../components/fixed_arith/fixed_adder_tree.sv",
        "../../../../components/fixed_arith/fixed_adder_tree_layer.sv",
        "../../../../components/fixed_arith/fixed_mult.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    extra_args.append(f"--unroll-count")
    extra_args.append(f"3000")
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_pvt",
        build_args=extra_args,
    )
    runner.test(
        hdl_toplevel="fixed_pvt",
        test_module="fixed_pvt_tb",
    )


if __name__ == "__main__":
    runner()
