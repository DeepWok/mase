import torch
import torch.nn as nn
import math

def tensor_cast(tensor_in, in_width, in_frac_width, out_width, out_frac_width):
    size = torch.tensor(tensor_in.shape)
    tensor_temp = tensor_in.reshape(torch.prod(size))

    for i in range(len(tensor_temp)):
        in_value = int(tensor_temp[i])
        if in_frac_width > out_frac_width:
            in_value = in_value >> (in_frac_width - out_frac_width)
        else:
            in_value = in_value << (out_frac_width - in_frac_width)
        in_int_width = in_width - in_frac_width
        out_int_width = out_width - out_frac_width
        if in_int_width > out_int_width:
            if in_value >> (in_frac_width + out_int_width) > 0:
                in_value = 1 << out_width - 1
            elif in_value >> (in_frac_width + out_int_width) < 0:
                in_value = -(1 << out_width - 1)
            else:
                in_value = int(in_value % (1 << out_width))
        tensor_temp[i] = in_value
    tensor_out = tensor_temp.reshape(list(size))
    # breakpoint()
    return tensor_out


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        weights,
        DWidth=32,
        DFWidth=8,
        WWidth=16,
        WFWidth=8,
    ):
        super(QuantizedLinear, self).__init__()

        self.QLinear = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            self.QLinear.weight = torch.nn.Parameter(weights)
        self.DWidth = DWidth
        self.DFWidth = DFWidth
        self.WWidth = WWidth
        self.WFWidth = WFWidth

    def forward(self, x):
        # Linear transformation with quantized weights
        output = self.QLinear(x)
        print("output = {}".format(output))
        # breakpoint()
        QuantizedOutput = tensor_cast(
            tensor_in=output,
            in_width=self.DWidth + self.WWidth + math.ceil(math.log2(x.shape[2])),
            in_frac_width=(self.DFWidth + self.WFWidth),
            out_width=self.DWidth,
            out_frac_width=self.DFWidth,
        )
        return QuantizedOutput


class QAttention(nn.Module):
    def __init__(
        self,
        dim,
        wq,
        wk,
        wv,
        DWidth=32,
        DFWidth=8,
        WWidth=16,
        WFWidth=8,
        num_heads=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.DWidth = DWidth
        self.DFWidth = DFWidth
        self.WWidth = WWidth
        self.WFWidth = WFWidth
        self.q = QuantizedLinear(
            dim,
            dim,
            weights=wq,
            DWidth=DWidth,
            DFWidth=DFWidth,
            WWidth=WWidth,
            WFWidth=WFWidth,
        )
        self.k = QuantizedLinear(
            dim,
            dim,
            weights=wk,
            DWidth=DWidth,
            DFWidth=DFWidth,
            WWidth=WWidth,
            WFWidth=WFWidth,
        )
        self.v = QuantizedLinear(
            dim,
            dim,
            weights=wv,
            DWidth=DWidth,
            DFWidth=DFWidth,
            WWidth=WWidth,
            WFWidth=WFWidth,
        )

    def forward(self, q_in,k_in,v_in):
        B, N, C = q_in.shape
        # breakpoint()
        q = (
            self.q(q_in)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(k_in)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(v_in)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        print("q = {} \n".format(q))
        print("k = {} \n".format(k))
        print("v = {} \n".format(v))
        attn = q @ k.transpose(-2, -1)

        qattn = tensor_cast(
            tensor_in=attn,
            in_width=self.DWidth + self.DWidth + math.ceil(math.log2(q.shape[2])),
            in_frac_width=(self.DFWidth + self.DFWidth),
            out_width=self.DWidth,
            out_frac_width=self.DFWidth,
        )
        print("qattn = {} \n".format(qattn))
        # breakpoint()
        x = (qattn @ v).transpose(1, 2).reshape(B, N, C)

        qx = tensor_cast(
            tensor_in=x,
            in_width=self.DWidth + self.DWidth + math.ceil(math.log2(qattn.shape[2])),
            in_frac_width=(self.DFWidth + self.DFWidth),
            out_width=self.DWidth,
            out_frac_width=self.DFWidth,
        )
        print("qx = {} \n".format(qx))
        return qx
