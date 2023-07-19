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
        DFWidth=1,
        WWidth=16,
        WFWidth=1,
        bias=True,
        bias_in=[],
        BWidth=16,
        BFWidth=1,
    ):
        super(QuantizedLinear, self).__init__()

        self.QLinear = nn.Linear(in_features, out_features, bias=bias)
        self.bias = bias
        self.DWidth = DWidth
        self.DFWidth = DFWidth
        self.WWidth = WWidth
        self.WFWidth = WFWidth
        self.BWidth = BWidth
        self.BFWidth = BFWidth
        bias_data = bias_in if bias else torch.zeros(out_features)
        bias_data = tensor_cast(
            tensor_in=bias_data,
            in_width=self.BWidth,
            in_frac_width=self.BFWidth,
            out_width=self.DWidth + self.WWidth,
            out_frac_width=self.DFWidth + self.WFWidth,
        )
        with torch.no_grad():
            self.QLinear.weight = torch.nn.Parameter(weights)
            self.QLinear.bias = torch.nn.Parameter(bias_data)

    def forward(self, x):
        # Linear transformation with quantized weights
        output = self.QLinear(x)
        in_width = (
            self.DWidth + self.WWidth + 1 if self.bias else self.DWidth + self.WWidth
        )
        QuantizedOutput = tensor_cast(
            tensor_in=output,
            in_width=in_width,
            in_frac_width=(self.DFWidth + self.WFWidth),
            out_width=self.DWidth,
            out_frac_width=self.DFWidth,
        )
        return QuantizedOutput


class QPartAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,  # always dim_out = dim / num_heads
        wqkv,
        bqkv,
        DWidth=32,
        DFWidth=1,
        WWidth=16,
        WFWidth=1,
        BWidth=16,
        BFWidth=1,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.DWidth = DWidth
        self.DFWidth = DFWidth
        self.WWidth = WWidth
        self.WFWidth = WFWidth
        print(
            "in_wqkv = {}\n\
        ".format(
                wqkv
            )
        )
        self.qkv = QuantizedLinear(
            dim,
            dim_out * 3,
            weights=wqkv,
            bias_in=bqkv,
            DWidth=DWidth,
            DFWidth=DFWidth,
            WWidth=WWidth,
            WFWidth=WFWidth,
            BWidth=BWidth,
            BFWidth=BFWidth,
        )

    def forward(self, q_in):
        B, N, C = q_in.shape
        qkv = self.qkv(q_in).reshape(B, N, 3, self.dim_out).permute(2, 0, 1, 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        print("queries = {} \n".format(queries))
        print("keys = {} \n".format(keys))
        print("values_t = {} \n".format(values.transpose(-2, -1)))
        energy = queries @ keys.transpose(-2, -1)
        qenergy = tensor_cast(
            tensor_in=energy,
            in_width=self.DWidth + self.DWidth + math.ceil(math.log2(queries.shape[2])),
            in_frac_width=(self.DFWidth + self.DFWidth),
            out_width=self.DWidth,
            out_frac_width=self.DFWidth,
        )
        print("qenergy = {} \n".format(qenergy))

        # from BHNC to BNHC to BN(HC)
        att = qenergy @ values
        qatt = tensor_cast(
            tensor_in=att,
            in_width=self.DWidth + self.DWidth + math.ceil(math.log2(qenergy.shape[2])),
            in_frac_width=(self.DFWidth + self.DFWidth),
            out_width=self.DWidth,
            out_frac_width=self.DFWidth,
        )
        print("qatt = {} \n".format(qatt))
        return qatt


class QHAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        wqkv,
        wp,
        bqkv,
        bp,
        DWidth=32,
        DFWidth=1,
        WWidth=16,
        WFWidth=1,
        BWidth=16,
        BFWidth=1,
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
        dim_out = int(dim / num_heads)
        wqkv = wqkv.reshape(num_heads, int(dim * 3 / num_heads), dim)
        bqkv = bqkv.reshape(num_heads, int(dim * 3 / num_heads))
        self.att_list = []
        for i in range(num_heads):
            self.qatt = QPartAttention(
                dim,
                dim_out,
                wqkv=wqkv[i],
                bqkv=bqkv[i],
                DWidth=DWidth,
                DFWidth=DFWidth,
                WWidth=WWidth,
                WFWidth=WFWidth,
                BWidth=BWidth,
                BFWidth=BFWidth,
            )
            self.att_list.append(self.qatt)

        self.projection = QuantizedLinear(
            dim,
            dim,
            wp,
            DWidth=DWidth,
            DFWidth=DFWidth,
            WWidth=WWidth,
            WFWidth=WFWidth,
            bias_in=bp,
            BWidth=BWidth,
            BFWidth=BFWidth,
        )

    def forward(self, q_in):
        result = self.att_list[0](q_in)
        for i in range(1, self.num_heads):
            other = self.att_list[i](q_in)
            result = torch.cat((result, other), 2)
        print("result = \n{}".format(result))

        out = self.projection(result)
        print("out = \n{}".format(out))
        return out
