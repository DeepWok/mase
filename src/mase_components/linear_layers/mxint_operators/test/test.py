from utils import mxint_quantize
import torch

block_size = 10
torch.manual_seed(0)
data = torch.rand(10)
w = torch.rand(10, 10)
d_man_width = 12
w_man_width = 8
e_width = 4
(data_in, mdata_in, edata_in) = mxint_quantize(
    data,
    d_man_width,
    e_width,
)
(weight, mweight, eweight) = mxint_quantize(
    w,
    w_man_width,
    e_width,
)
linear = torch.nn.Linear(10, 10, bias=False)
linear.weight = torch.nn.Parameter(weight)
target_x = linear(data_in)
linear.weight = torch.nn.Parameter(mweight)
hardware_out = linear(mdata_in)
print(hardware_out * (2 ** (edata_in + eweight)))
# software knows
print(target_x)


# hardware quant back
def hardware_quant(hardware_in, be_value, e_width, width):
    from math import ceil, log2

    result = ceil(log2(max(hardware_in)))
    exponent_bias = 2 ** (e_width - 1) - 1

    # exponent
    exponent_max = 2**e_width - 1 - exponent_bias
    exponent_min = -exponent_bias
    exponent = (
        torch.ceil(torch.log2(hardware_in.abs().max())) + be_value - exponent_bias
    )
    exponent = torch.clamp(exponent, exponent_min, exponent_max)
    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
    mantissa = hardware_in / 2 ** (exponent - be_value)
    breakpoint()
    mantissa = torch.clamp(mantissa.floor(), int_min, int_max)

    msfp_x = (2**exponent) * mantissa
    return msfp_x, mantissa, exponent


new_man_width = 8
new_e_width = 4
qout, qmout, qeout = hardware_quant(
    hardware_out, (edata_in + eweight), new_e_width, new_man_width
)
out, mout, eout = mxint_quantize(target_x, new_man_width, new_e_width)
breakpoint()
# def hardware_quant_back():
#    hardware_out.max().log2()+ hardware_exp
# clamp((log2(max(hardware_out))+hardware_exp),target_width)
