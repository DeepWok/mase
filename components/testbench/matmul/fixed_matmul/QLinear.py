import torch
import torch.nn as nn


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


class QuantizedMatmulBias(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        weights,
        DWidth=32,
        DFWidth=1,
        WWidth=16,
        WFWidth=1,
        bias=False,
        bias_in=[],
        BWidth=16,
        BFWidth=1,
    ):
        super(QuantizedMatmulBias, self).__init__()

        self.QLinear = nn.Linear(in_features, out_features, bias=False)
        self.bias = bias
        self.DWidth = DWidth
        self.DFWidth = DFWidth
        self.WWidth = WWidth
        self.WFWidth = WFWidth
        self.BWidth = BWidth
        self.BFWidth = BFWidth

        self.weight = weights
        bias_data = bias_in if bias else torch.zeros(out_features)
        self.bias_data = tensor_cast(
            tensor_in=bias_data,
            in_width=self.BWidth,
            in_frac_width=self.BFWidth,
            out_width=self.DWidth + self.WWidth,
            out_frac_width=self.DFWidth + self.WFWidth,
        )
        with torch.no_grad():
            self.QLinear.weight = torch.nn.Parameter(weights)

    def forward(self, x):
        # Linear transformation with quantized weights
        output = self.QLinear(x) + self.bias_data if self.bias else self.QLinear(x)

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
