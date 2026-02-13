import torch
import torch.nn as nn
import math
from math import log2
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from z_qlayers.qlinear import QuantizedLinear
from z_qlayers.qconv import QuantizedConvolution
from z_qlayers.qatt import QPartAttention
from z_qlayers.qmm import QuantizedMatmulBias
from z_qlayers.tensor_cast import tensor_cast


class QOverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        weights,
        bias,
        img_size=224,
        patch_size=7,
        stride=4,
        in_chans=3,
        embed_dim=768,
        data_width=32,
        data_frac_width=8,
        weight_width=16,
        weight_frac_width=8,
        bias_width=32,
        bias_frac_width=8,
        out_width=32,
        out_frac_width=1,
    ):
        super().__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        assert max(patch_size) >= stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = QuantizedConvolution(
            in_chans,
            embed_dim,
            patch_size,
            weights,
            bias,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
            data_width=data_width,
            data_frac_width=data_frac_width,
            weight_width=weight_width,
            weight_frac_width=weight_frac_width,
            bias_width=bias_width,
            bias_frac_width=bias_frac_width,
            out_width=out_width,
            out_frac_width=out_frac_width,
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class QuantizedMSA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        wqkv,
        wp,
        bqkv,
        bp,
        data_width=32,
        data_frac_width=1,
        weight_q_width=16,
        weight_q_frac_width=1,
        weight_k_width=16,
        weight_k_frac_width=1,
        weight_v_width=16,
        weight_v_frac_width=1,
        weight_p_width=16,
        weight_p_frac_width=1,
        bias_q_width=16,
        bias_q_frac_width=1,
        bias_k_width=16,
        bias_k_frac_width=1,
        bias_v_width=16,
        bias_v_frac_width=1,
        bias_p_width=16,
        bias_p_frac_width=1,
        data_q_width=16,
        data_q_frac_width=1,
        data_k_width=16,
        data_k_frac_width=1,
        data_v_width=16,
        data_v_frac_width=1,
        data_s_width=16,
        data_s_frac_width=1,
        data_z_width=16,
        data_z_frac_width=1,
        out_width=32,
        out_frac_width=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        dim_out = int(dim / num_heads)
        wqkv = wqkv.reshape(num_heads, int(dim * 3 / num_heads), dim)
        bqkv = bqkv.reshape(num_heads, int(dim * 3 / num_heads))
        self.att_list = []
        for i in range(num_heads):
            self.qatt = QPartAttention(
                dim,
                dim_out,
                wqkv[i],
                bqkv[i],
                data_width,
                data_frac_width,
                weight_q_width,
                weight_q_frac_width,
                weight_k_width,
                weight_k_frac_width,
                weight_v_width,
                weight_v_frac_width,
                bias_q_width,
                bias_q_frac_width,
                bias_k_width,
                bias_k_frac_width,
                bias_v_width,
                bias_v_frac_width,
                data_q_width,
                data_q_frac_width,
                data_k_width,
                data_k_frac_width,
                data_v_width,
                data_v_frac_width,
                data_s_width,
                data_s_frac_width,
                data_z_width,
                data_z_frac_width,
            )
            self.att_list.append(self.qatt)

        self.projection = QuantizedLinear(
            dim,
            dim,
            wp,
            data_width=data_z_width,
            data_frac_width=data_z_frac_width,
            weight_width=weight_p_width,
            weight_frac_width=weight_p_frac_width,
            bias_in=bp,
            bias_width=bias_p_width,
            bias_frac_width=bias_p_frac_width,
            out_width=out_width,
            out_frac_width=out_frac_width,
        )

    def forward(self, q_in):
        result = self.att_list[0](q_in)
        for i in range(1, self.num_heads):
            other = self.att_list[i](q_in)
            result = torch.cat((result, other), 2)
        print("result = ", result)
        out = self.projection(result)
        return out


class QuantizedMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        weights1,
        weights2,
        in_features,
        hidden_features,
        bias=True,
        bias_in1=[],
        bias_in2=[],
        data_width=32,
        data_frac_width=8,
        weight_i2h_width=16,
        weight_i2h_frac_width=8,
        weight_h2o_width=16,
        weight_h2o_frac_width=8,
        bias_i2h_width=32,
        bias_i2h_frac_width=8,
        bias_h2o_width=32,
        bias_h2o_frac_width=8,
        hidden_width=32,
        hidden_frac_width=2,
        out_width=32,
        out_frac_width=1,
    ):
        super().__init__()
        self.fc1 = QuantizedLinear(
            in_features,
            hidden_features,
            weights=weights1,
            bias=bias,
            bias_in=bias_in1,
            data_width=data_width,
            data_frac_width=data_frac_width,
            weight_width=weight_i2h_width,
            weight_frac_width=weight_i2h_frac_width,
            bias_width=bias_i2h_width,
            bias_frac_width=bias_i2h_frac_width,
            out_width=hidden_width,
            out_frac_width=hidden_frac_width,
        )
        # dont need it in the integer mode
        # self.act = act_layer()
        self.fc2 = QuantizedLinear(
            hidden_features,
            in_features,
            weights=weights2,
            bias=bias,
            bias_in=bias_in2,
            data_width=hidden_width,
            data_frac_width=hidden_frac_width,
            weight_width=weight_h2o_width,
            weight_frac_width=weight_h2o_frac_width,
            bias_width=bias_h2o_width,
            bias_frac_width=bias_h2o_frac_width,
            out_width=out_width,
            out_frac_width=out_frac_width,
        )

    def forward(self, x):
        x = self.fc1(x)
        # x = self.act(x)
        x = self.fc2(x)
        return x


class QuantizedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        wqkv,
        wp,
        bqkv,
        bp,
        weights1,
        weights2,
        in_features,
        hidden_features,
        bias=True,
        bias_in1=[],
        bias_in2=[],
        in_width=32,
        in_frac_width=1,
        weight_q_width=16,
        weight_q_frac_width=1,
        weight_k_width=16,
        weight_k_frac_width=1,
        weight_v_width=16,
        weight_v_frac_width=1,
        weight_p_width=16,
        weight_p_frac_width=1,
        bias_q_width=16,
        bias_q_frac_width=1,
        bias_k_width=16,
        bias_k_frac_width=1,
        bias_v_width=16,
        bias_v_frac_width=1,
        bias_p_width=16,
        bias_p_frac_width=1,
        data_q_width=16,
        data_q_frac_width=1,
        data_k_width=16,
        data_k_frac_width=1,
        data_v_width=16,
        data_v_frac_width=1,
        data_s_width=16,
        data_s_frac_width=1,
        data_z_width=16,
        data_z_frac_width=1,
        msa_out_width=16,
        msa_out_frac_width=1,
        weight_i2h_width=16,
        weight_i2h_frac_width=8,
        weight_h2o_width=16,
        weight_h2o_frac_width=8,
        bias_i2h_width=32,
        bias_i2h_frac_width=8,
        bias_h2o_width=32,
        bias_h2o_frac_width=8,
        hidden_width=32,
        hidden_frac_width=2,
        out_width=32,
        out_frac_width=1,
    ):
        super().__init__()
        self.qmsa = QuantizedMSA(
            dim,
            num_heads,
            wqkv,
            wp,
            bqkv,
            bp,
            in_width,
            in_frac_width,
            weight_q_width,
            weight_q_frac_width,
            weight_k_width,
            weight_k_frac_width,
            weight_v_width,
            weight_v_frac_width,
            weight_p_width,
            weight_p_frac_width,
            bias_q_width,
            bias_q_frac_width,
            bias_k_width,
            bias_k_frac_width,
            bias_v_width,
            bias_v_frac_width,
            bias_p_width,
            bias_p_frac_width,
            data_q_width,
            data_q_frac_width,
            data_k_width,
            data_k_frac_width,
            data_v_width,
            data_v_frac_width,
            data_s_width,
            data_s_frac_width,
            data_z_width,
            data_z_frac_width,
            msa_out_width,
            msa_out_frac_width,
        )
        # dont need it in the integer mode
        # self.act = act_layer()
        self.qmlp = QuantizedMlp(
            weights1,
            weights2,
            in_features,
            hidden_features,
            bias,
            bias_in1,
            bias_in2,
            msa_out_width + 1,
            msa_out_frac_width,
            weight_i2h_width,
            weight_i2h_frac_width,
            weight_h2o_width,
            weight_h2o_frac_width,
            bias_i2h_width,
            bias_i2h_frac_width,
            bias_h2o_width,
            bias_h2o_frac_width,
            hidden_width,
            hidden_frac_width,
            out_width - 1,
            out_frac_width,
        )
        self.in_width = in_width
        self.in_frac_width = in_frac_width
        self.msa_out_width = msa_out_width
        self.msa_out_frac_width = msa_out_frac_width
        self.out_width = out_width
        self.out_frac_width = out_frac_width

    def forward(self, x):
        qmsa_x = self.qmsa(x)
        res_msa_x = tensor_cast(
            tensor_in=x,
            in_width=self.in_width,
            in_frac_width=self.in_frac_width,
            out_width=self.msa_out_width,
            out_frac_width=self.msa_out_frac_width,
        )
        x = qmsa_x + res_msa_x

        qmlp_x = self.qmlp(x)
        res_mlp_x = tensor_cast(
            tensor_in=x,
            in_width=self.msa_out_width,
            in_frac_width=self.msa_out_frac_width,
            out_width=self.out_width - 1,
            out_frac_width=self.out_frac_width,
        )
        x = qmlp_x + res_mlp_x
        return x
