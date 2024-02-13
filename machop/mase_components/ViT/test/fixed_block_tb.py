#!/usr/bin/env python3

import random, os, math, logging, sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random, os, math, logging, sys
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# from torchsummary import summary
from einops import rearrange, reduce, repeat


from random_test import RandomSource
from random_test import RandomSink
from random_test import check_results

import cocotb
from cocotb.triggers import Timer
from cocotb.triggers import FallingEdge
from cocotb.clock import Clock
from cocotb.runner import get_runner

from pvt_quant import QuantizedBlock
from ha_softmax import generate_table_hardware, generate_table_div_hardware
from z_qlayers import quantize_to_int as q2i
from z_qlayers import linear_data_pack

debug = True

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


# DUT test specifications
class VerificationCase:
    def __init__(self, samples=1):
        self.samples = samples
        self.width_generate()

        self.has_bias = 1
        self.in_num = 2
        self.in_dim = 12
        self.num_heads = 2
        self.wqkv_dim = self.in_dim / self.num_heads
        self.wp_dim = self.in_dim
        self.unroll_in_num = 1
        self.unroll_in_dim = 2
        self.unroll_wqkv_dim = 3

        self.in_parallelism = self.unroll_in_num
        self.in_num_parallelism = int(self.in_num / self.unroll_in_num)

        self.in_size = self.unroll_in_dim
        self.in_depth = int(self.in_dim / self.unroll_in_dim)

        # noted num_heads * wqkv_p * wqkv_np should be = in_s * in_d
        self.wqkv_parallelism = self.unroll_wqkv_dim
        self.wqkv_num_parallelism = int(self.wqkv_dim / self.unroll_wqkv_dim)

        self.wp_parallelism = self.in_size
        self.wp_num_parallelism = self.in_depth
        assert (
            self.num_heads * self.wqkv_parallelism * self.wqkv_num_parallelism
            == self.in_size * self.in_depth
        ), "should have num_heads * wqkv_p * wqkv_np == in_s * in_d"
        assert (
            (self.in_num % self.unroll_in_num == 0)
            and (self.in_dim % self.unroll_in_dim == 0)
            and (self.wqkv_dim % self.unroll_wqkv_dim == 0)
        ), "unroll parameter should be exact division of all"

        self.wp_size = self.num_heads * self.wqkv_parallelism
        self.wp_depth = self.wqkv_num_parallelism

        self.in_num = self.in_num
        self.in_features = self.in_dim
        self.mlp_ratio = 2
        self.hidden_features = self.mlp_ratio * self.in_features
        self.out_features = self.in_features

        self.unroll_in_num = self.unroll_in_num
        self.unroll_in_features = self.unroll_in_dim
        self.unroll_hidden_features = 2
        self.unroll_out_features = self.unroll_in_features
        # data_generate
        self.source_generate()
        ## remain modification
        self.outputs = RandomSink(
            samples=samples * self.in_num_parallelism * self.wp_num_parallelism,
            max_stalls=2 * samples * self.in_num_parallelism * self.wp_parallelism,
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
            "head_proj": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 5,
                "weight_width": 8,
                "weight_frac_width": 6,
                "bias_width": 8,
                "bias_frac_width": 4,
            },
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
                    "data_in_width": mlp_o,
                    "data_in_frac_width": mlp_o_f,
                },
            },
            "pvt_norm": {
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
        }
        self.ow, self.ow_f = mlp_o + 1, mlp_o_f

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.w_config["block"]["affine_att"]["mul"]["data_in_width"],
            "IN_FRAC_WIDTH": self.w_config["block"]["affine_att"]["mul"][
                "data_in_frac_width"
            ],
            "AF_MSA_ADD_WIDTH": self.w_config["block"]["affine_att"]["add"][
                "data_in_width"
            ],
            "AF_MSA_ADD_FRAC_WIDTH": self.w_config["block"]["affine_att"]["add"][
                "data_in_frac_width"
            ],
            "MSA_IN_WIDTH": self.w_config["block"]["msa"]["q_proj"]["data_in_width"],
            "MSA_IN_FRAC_WIDTH": self.w_config["block"]["msa"]["q_proj"][
                "data_in_frac_width"
            ],
            "WQ_WIDTH": self.w_config["block"]["msa"]["q_proj"]["weight_width"],
            "WQ_FRAC_WIDTH": self.w_config["block"]["msa"]["q_proj"][
                "weight_frac_width"
            ],
            "WK_WIDTH": self.w_config["block"]["msa"]["kv_proj"]["weight_width"],
            "WK_FRAC_WIDTH": self.w_config["block"]["msa"]["kv_proj"][
                "weight_frac_width"
            ],
            "WV_WIDTH": self.w_config["block"]["msa"]["kv_proj"]["weight_width"],
            "WV_FRAC_WIDTH": self.w_config["block"]["msa"]["kv_proj"][
                "weight_frac_width"
            ],
            "WP_WIDTH": self.w_config["block"]["msa"]["z_proj"]["weight_width"],
            "WP_FRAC_WIDTH": self.w_config["block"]["msa"]["z_proj"][
                "weight_frac_width"
            ],
            "BQ_WIDTH": self.w_config["block"]["msa"]["q_proj"]["bias_width"],
            "BQ_FRAC_WIDTH": self.w_config["block"]["msa"]["q_proj"]["bias_frac_width"],
            "BK_WIDTH": self.w_config["block"]["msa"]["kv_proj"]["bias_width"],
            "BK_FRAC_WIDTH": self.w_config["block"]["msa"]["kv_proj"][
                "bias_frac_width"
            ],
            "BV_WIDTH": self.w_config["block"]["msa"]["kv_proj"]["bias_width"],
            "BV_FRAC_WIDTH": self.w_config["block"]["msa"]["kv_proj"][
                "bias_frac_width"
            ],
            "BP_WIDTH": self.w_config["block"]["msa"]["z_proj"]["bias_width"],
            "BP_FRAC_WIDTH": self.w_config["block"]["msa"]["z_proj"]["bias_frac_width"],
            "DQ_WIDTH": self.w_config["block"]["msa"]["attn_matmul"]["data_in_width"],
            "DQ_FRAC_WIDTH": self.w_config["block"]["msa"]["attn_matmul"][
                "data_in_frac_width"
            ],
            "DK_WIDTH": self.w_config["block"]["msa"]["attn_matmul"]["weight_width"],
            "DK_FRAC_WIDTH": self.w_config["block"]["msa"]["attn_matmul"][
                "weight_frac_width"
            ],
            "DS_WIDTH": self.w_config["block"]["msa"]["z_matmul"]["data_in_width"],
            "DS_FRAC_WIDTH": self.w_config["block"]["msa"]["z_matmul"][
                "data_in_frac_width"
            ],
            "DV_WIDTH": self.w_config["block"]["msa"]["z_matmul"]["weight_width"],
            "DV_FRAC_WIDTH": self.w_config["block"]["msa"]["z_matmul"][
                "weight_frac_width"
            ],
            "EXP_WIDTH": self.w_config["block"]["msa"]["softmax"]["exp_width"],
            "EXP_FRAC_WIDTH": self.w_config["block"]["msa"]["softmax"][
                "exp_frac_width"
            ],
            "DIV_WIDTH": self.w_config["block"]["msa"]["softmax"]["div_width"],
            "DS_SOFTMAX_WIDTH": self.w_config["block"]["msa"]["softmax"][
                "data_out_width"
            ],
            "DS_SOFTMAX_FRAC_WIDTH": self.w_config["block"]["msa"]["softmax"][
                "data_out_frac_width"
            ],
            "DZ_WIDTH": self.w_config["block"]["msa"]["z_proj"]["data_in_width"],
            "DZ_FRAC_WIDTH": self.w_config["block"]["msa"]["z_proj"][
                "data_in_frac_width"
            ],
            "AF_MLP_IN_WIDTH": self.w_config["block"]["affine_mlp"]["mul"][
                "data_in_width"
            ],
            "AF_MLP_IN_FRAC_WIDTH": self.w_config["block"]["affine_mlp"]["mul"][
                "data_in_frac_width"
            ],
            "AF_MLP_ADD_WIDTH": self.w_config["block"]["affine_mlp"]["add"][
                "data_in_width"
            ],
            "AF_MLP_ADD_FRAC_WIDTH": self.w_config["block"]["affine_mlp"]["add"][
                "data_in_frac_width"
            ],
            # mlp
            "MLP_IN_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"]["data_in_width"],
            "MLP_IN_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "data_in_frac_width"
            ],
            "WEIGHT_I2H_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "weight_width"
            ],
            "WEIGHT_I2H_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "weight_frac_width"
            ],
            "BIAS_I2H_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"]["bias_width"],
            "BIAS_I2H_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc1_proj"][
                "bias_frac_width"
            ],
            "MLP_HAS_BIAS": self.has_bias,
            "HIDDEN_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"]["data_in_width"],
            "HIDDEN_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "data_in_frac_width"
            ],
            "WEIGHT_H2O_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "weight_width"
            ],
            "WEIGHT_H2O_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "weight_frac_width"
            ],
            "BIAS_H2O_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"]["bias_width"],
            "BIAS_H2O_FRAC_WIDTH": self.w_config["block"]["mlp"]["fc2_proj"][
                "bias_frac_width"
            ],
            "OUT_WIDTH": self.ow,
            "OUT_FRAC_WIDTH": self.ow_f,
            "IN_NUM": self.in_num,
            "IN_DIM": self.in_dim,
            "MLP_RATIO": self.mlp_ratio,
            "NUM_HEADS": self.num_heads,
            "UNROLL_IN_NUM": self.unroll_in_num,
            "UNROLL_IN_DIM": self.unroll_in_dim,
            "UNROLL_WQKV_DIM": self.unroll_wqkv_dim,
            "UNROLL_HIDDEN_FEATURES": self.unroll_hidden_features,
        }

    def source_generate(self):
        samples = self.samples
        torch.manual_seed(2)
        self.x = 3 * torch.randn((samples, self.in_num, self.in_dim))
        w_config = self.w_config["block"]
        self.block = QuantizedBlock(
            self.in_dim,
            self.num_heads,
            w_config,
            mlp_ratio=self.hidden_features / self.in_features,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
        )
        input_tensor = q2i(
            self.x,
            w_config["msa"]["q_proj"]["data_in_width"],
            w_config["msa"]["q_proj"]["data_in_frac_width"],
        )
        self.data_in = self.data_pack(
            input_tensor,
            self.in_num_parallelism,
            self.in_depth,
            self.in_parallelism,
            self.in_size,
        )
        self.data_in.reverse()
        self.inputs = RandomSource(
            name="data_in",
            samples=samples * self.in_depth * self.in_num_parallelism,
            num=self.in_parallelism * self.in_size,
            max_stalls=2 * samples * self.in_depth * self.in_num_parallelism,
            data_specify=self.data_in,
            debug=debug,
        )
        att = self.block.attn
        self.msa_data_generate(att)
        mlp = self.block.mlp
        self.mlp_data_generate(mlp)

        aff_att = self.block.norm1
        num = self.in_num_parallelism * self.in_depth
        in_size = self.in_parallelism * self.in_size
        aff_att_w, aff_att_b = self.aff_data_generate(
            w_config["affine_att"], aff_att, num, in_size
        )
        aff_mlp = self.block.norm2
        aff_mlp_w, aff_mlp_b = self.aff_data_generate(
            w_config["affine_mlp"], aff_mlp, num, in_size
        )
        # breakpoint()
        self.aff_att_weight = RandomSource(
            samples=samples * num,
            max_stalls=2 * samples,
            num=in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=aff_att_w,
        )
        self.aff_att_bias = RandomSource(
            samples=samples * num,
            max_stalls=2 * samples,
            num=in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=aff_att_b,
        )

        self.aff_mlp_weight = RandomSource(
            samples=samples * num,
            max_stalls=2 * samples,
            num=in_size,
            is_data_vector=True,
            debug=debug,
            data_specify=aff_mlp_w,
        )
        self.aff_mlp_bias = RandomSource(
            samples=samples * num,
            max_stalls=2 * samples,
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

    def msa_data_generate(self, qatt):
        # generate data
        samples = self.samples
        config = self.w_config["block"]["msa"]
        in_x = self.in_dim
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
        in_depth = self.in_depth
        in_size = self.in_size
        wqkv_parallelism = self.wqkv_parallelism
        wqkv_num_parallelism = self.wqkv_num_parallelism
        num_heads = self.num_heads
        wp_parallelism = self.wp_parallelism
        wp_num_parallelism = self.wp_num_parallelism
        wp_depth = wqkv_num_parallelism
        wp_size = num_heads * wqkv_parallelism
        dim = in_size * in_depth

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
            name="weight_q",
            samples=samples * in_depth * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism * in_size,
            max_stalls=2 * samples * in_depth * wqkv_num_parallelism,
            data_specify=wq_in,
            debug=debug,
        )
        self.weight_k = RandomSource(
            name="weight_k",
            samples=samples * in_depth * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism * in_size,
            max_stalls=2 * samples * in_depth * wqkv_num_parallelism,
            data_specify=wk_in,
            debug=debug,
        )
        self.weight_v = RandomSource(
            name="weight_v",
            samples=samples * in_depth * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism * in_size,
            max_stalls=2 * samples * in_depth * wqkv_num_parallelism,
            data_specify=wv_in,
            debug=debug,
        )
        self.weight_p = RandomSource(
            name="weight_p",
            samples=samples * wp_depth * wp_num_parallelism,
            num=wp_parallelism * wp_size,
            max_stalls=2 * samples * wp_depth * wp_num_parallelism,
            data_specify=wp_in,
            debug=debug,
        )
        self.bias_q = RandomSource(
            name="bias_q",
            samples=samples * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=bq_in,
            debug=debug,
        )
        self.bias_k = RandomSource(
            name="bias_k",
            samples=samples * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=bk_in,
            debug=debug,
        )
        self.bias_v = RandomSource(
            name="bias_v",
            samples=samples * wqkv_num_parallelism,
            num=num_heads * wqkv_parallelism,
            max_stalls=2 * samples,
            data_specify=bv_in,
            debug=debug,
        )
        self.bias_p = RandomSource(
            name="bias_p",
            samples=samples * wp_num_parallelism,
            num=wp_parallelism,
            max_stalls=2 * samples,
            data_specify=bp_in,
            debug=debug,
        )

    def mlp_data_generate(self, qmlp):
        samples = self.samples
        w_config = self.w_config["block"]["mlp"]
        in_features = self.in_features
        hidden_features = self.hidden_features
        out_features = self.out_features
        unroll_in_features = self.unroll_in_features
        unroll_hidden_features = self.unroll_hidden_features
        unroll_out_features = self.unroll_out_features
        depth_in_features = in_features // unroll_in_features
        depth_hidden_features = hidden_features // unroll_hidden_features
        depth_out_features = out_features // unroll_out_features
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
            hidden_features,
            in_features,
            unroll_hidden_features,
            unroll_in_features,
        )
        bias1_in = linear_data_pack(
            samples,
            bias1_tensor.repeat(samples, 1, 1),
            hidden_features,
            1,
            unroll_hidden_features,
            1,
        )
        weight2_in = linear_data_pack(
            samples,
            weight2_tensor.repeat(samples, 1, 1),
            out_features,
            hidden_features,
            unroll_out_features,
            unroll_hidden_features,
        )
        bias2_in = linear_data_pack(
            samples,
            bias2_tensor.repeat(samples, 1, 1),
            out_features,
            1,
            unroll_out_features,
            1,
        )
        weight1_in.reverse()
        bias1_in.reverse()
        weight2_in.reverse()
        bias2_in.reverse()
        self.bias1 = RandomSource(
            name="bias1",
            samples=samples * depth_hidden_features,
            num=unroll_hidden_features,
            max_stalls=2 * samples * depth_hidden_features,
            data_specify=bias1_in,
            debug=debug,
        )
        self.bias2 = RandomSource(
            name="bias2",
            samples=samples * depth_out_features,
            num=unroll_out_features,
            max_stalls=2 * samples * depth_out_features,
            data_specify=bias2_in,
            debug=debug,
        )
        self.weight1 = RandomSource(
            name="weight1",
            samples=samples * depth_hidden_features * depth_in_features,
            num=unroll_hidden_features * unroll_in_features,
            max_stalls=2 * samples * depth_hidden_features * depth_in_features,
            data_specify=weight1_in,
            debug=debug,
        )
        self.weight2 = RandomSource(
            name="weight2",
            samples=samples * depth_out_features * depth_hidden_features,
            num=unroll_out_features * unroll_hidden_features,
            max_stalls=2 * samples * depth_hidden_features * depth_out_features,
            data_specify=weight2_in,
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

    def sw_compute(self):
        output = self.block(self.x)
        out_data = self.data_pack(
            q2i(output, self.ow, self.ow_f),
            self.in_num_parallelism,
            self.out_features // self.unroll_out_features,
            self.in_parallelism,
            self.unroll_out_features,
        )
        return out_data


def wave_check(dut):
    logger.debug(
        "wave of in_out:\n\
            {},{},data_in = {} \n\
            {},{},af_msa = {} \n\
            {},{},msa_out = {} \n\
            {},{},res_msa = {} \n\
            {},{},af_mlp = {} \n\
            {},{},mlp_out = {} \n\
            {},{},data_out = {}\n\
            ".format(
            dut.data_in_valid.value,
            dut.data_in_ready.value,
            [int(i) for i in dut.data_in.value],
            dut.af_msa_out_valid.value,
            dut.af_msa_out_ready.value,
            [int(i) for i in dut.af_msa_out.value],
            dut.msa_out_valid.value,
            dut.msa_out_ready.value,
            [int(i) for i in dut.msa_out.value],
            dut.res_msa_valid.value,
            dut.res_msa_ready.value,
            [int(i) for i in dut.res_msa.value],
            dut.af_mlp_out_valid.value,
            dut.af_mlp_out_ready.value,
            [int(i) for i in dut.af_mlp_out.value],
            dut.mlp_out_valid.value,
            dut.mlp_out_ready.value,
            [int(i) for i in dut.mlp_out.value],
            dut.data_out_valid.value,
            dut.data_out_ready.value,
            [int(i) for i in dut.data_out.value],
        )
    )


@cocotb.test()
async def test_att(dut):
    """Test integer based vector mult"""
    samples = 20
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
    dut.weight_q_valid.value = 0
    dut.weight_k_valid.value = 0
    dut.weight_v_valid.value = 0
    dut.weight_p_valid.value = 0
    dut.bias_q_valid.value = 0
    dut.bias_k_valid.value = 0
    dut.bias_v_valid.value = 0
    dut.bias_p_valid.value = 0
    dut.data_in_valid.value = 0
    dut.data_out_ready.value = 1
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    done = False
    count_af_msa = 0
    count_msa = 0
    count_mlp = 0
    count_mlp_hidden = 0
    count_mlp_out = 0
    # Set a timeout to avoid deadlock
    for i in range(samples * 6000):
        await FallingEdge(dut.clk)
        # breakpoint()
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
        dut.data_in_valid.value = test_case.inputs.pre_compute()

        dut.af_mlp_weight_valid.value = test_case.aff_mlp_weight.pre_compute()
        dut.af_mlp_bias_valid.value = test_case.aff_mlp_bias.pre_compute()
        dut.weight_in2hidden_valid.value = test_case.weight1.pre_compute()
        dut.bias_in2hidden_valid.value = test_case.bias1.pre_compute()
        dut.weight_hidden2out_valid.value = test_case.weight2.pre_compute()
        dut.bias_hidden2out_valid.value = test_case.bias2.pre_compute()
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.pre_compute(
            dut.data_out_valid.value
        )
        await Timer(1, units="ns")
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

        dut.data_in_valid.value, dut.data_in.value = test_case.inputs.compute(
            dut.data_in_ready.value
        )
        await Timer(1, units="ns")
        dut.data_out_ready.value = test_case.outputs.compute(
            dut.data_out_valid.value, dut.data_out.value
        )
        await Timer(1, units="ns")
        if dut.af_msa_out_valid.value == 1 and dut.af_msa_out_ready.value == 1:
            count_af_msa += 1
        if dut.msa_out_valid.value == 1 and dut.msa_out_ready.value == 1:
            count_msa += 1
        if (
            dut.mlp_inst.hidden_data_valid.value == 1
            and dut.mlp_inst.hidden_data_ready.value == 1
        ):
            count_mlp_hidden += 1
        if dut.mlp_out_valid.value == 1 and dut.mlp_out_ready.value == 1:
            count_mlp += 1

        print("count_af_msa = ", count_af_msa)
        print("count_msa = ", count_msa)
        print("count_mlp_hidden = ", count_mlp_hidden)
        print("count_mlp = ", count_mlp)
        wave_check(dut)
        if (
            test_case.weight1.is_empty()
            and test_case.bias1.is_empty()
            and test_case.weight2.is_empty()
            and test_case.bias2.is_empty()
            and test_case.weight_q.is_empty()
            and test_case.weight_k.is_empty()
            and test_case.weight_v.is_empty()
            and test_case.weight_p.is_empty()
            and test_case.bias_q.is_empty()
            and test_case.bias_k.is_empty()
            and test_case.bias_v.is_empty()
            and test_case.bias_p.is_empty()
            and test_case.inputs.is_empty()
            and test_case.outputs.is_full()
        ):
            done = True
            break
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"

    check_results(test_case.outputs.data, test_case.ref)


def runner():
    sim = os.getenv("SIM", "verilator")

    verilog_sources = [
        "../../../../components/ViT/fixed_block.sv",
        "../../../../components/ViT/hash_softmax.sv",
        "../../../../components/ViT/affine_layernorm.sv",
        "../../../../components/ViT/fixed_mlp.sv",
        "../../../../components/ViT/fixed_msa.sv",
        "../../../../components/attention/fixed_self_att.sv",
        "../../../../components/attention/fixed_att.sv",
        "../../../../components/conv/roller.sv",
        "../../../../components/common/fifo.sv",
        "../../../../components/common/unpacked_fifo.sv",
        "../../../../components/common/input_buffer.sv",
        "../../../../components/common/blk_mem_gen_0.sv",
        "../../../../components/common/skid_buffer.sv",
        "../../../../components/common/unpacked_skid_buffer.sv",
        "../../../../components/common/register_slice.sv",
        "../../../../components/common/split2.sv",
        "../../../../components/common/join2.sv",
        "../../../../components/matmul/fixed_matmul.sv",
        "../../../../components/linear/fixed_linear.sv",
        "../../../../components/linear/fixed_2d_linear.sv",
        "../../../../components/cast/fixed_rounding.sv",
        "../../../../components/activations/fixed_relu.sv",
        "../../../../components/fixed_arithmetic/fixed_matmul_core.sv",
        "../../../../components/fixed_arithmetic/fixed_dot_product.sv",
        "../../../../components/fixed_arithmetic/fixed_accumulator.sv",
        "../../../../components/fixed_arithmetic/fixed_vector_mult.sv",
        "../../../../components/fixed_arithmetic/fixed_adder_tree.sv",
        "../../../../components/fixed_arithmetic/fixed_adder_tree_layer.sv",
        "../../../../components/fixed_arithmetic/fixed_mult.sv",
    ]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_block",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="fixed_block", test_module="fixed_block_tb")


if __name__ == "__main__":
    runner()
