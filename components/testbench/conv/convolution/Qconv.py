import torch
import torch.nn as nn
from math import log2


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


class QuantizedConvolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        weights,
        bias_data,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        DWidth=32,
        DFWidth=8,
        WWidth=16,
        WFWidth=8,
        BWidth=32,
        BFWidth=8,
    ):
        super(QuantizedConvolution, self).__init__()
        self.kernel_0, self.kernel_1 = kernel_size
        self.QConv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
            dtype=None,
        )
        self.in_channels = in_channels
        self.DWidth = DWidth
        self.DFWidth = DFWidth
        self.WWidth = WWidth
        self.WFWidth = WFWidth
        self.BWidth = BWidth
        self.BFWidth = BFWidth
        bias_data = tensor_cast(
            tensor_in=bias_data,
            in_width=BWidth,
            in_frac_width=BFWidth,
            out_width=self.DWidth
            + self.WWidth
            + log2(self.kernel_0 * self.kernel_1 * self.in_channels),
            out_frac_width=(self.DFWidth + self.WFWidth),
        )
        with torch.no_grad():
            self.QConv.weight = torch.nn.Parameter(weights)
            self.QConv.bias = torch.nn.Parameter(bias_data)

        # print(self.QConv.bias.shape)
        # print(self.QConv.weight.shape)

    def forward(self, x):
        # Linear transformation with quantized weights
        output = self.QConv(x)
        print(output)
        QuantizedOutput = tensor_cast(
            tensor_in=output,
            in_width=self.DWidth
            + self.WWidth
            + int(log2(self.kernel_0 * self.kernel_1 * self.in_channels))
            + 1,
            in_frac_width=(self.DFWidth + self.WFWidth),
            out_width=self.DWidth,
            out_frac_width=self.DFWidth,
        )
        return QuantizedOutput
