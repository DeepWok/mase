import torch
from chop.nn.quantized.modules import (
    ConvTranspose2dInteger,
    ConvTranspose2dBinary,
    ConvTranspose2dMinifloatIEEE,
)

def test_convtranspose2d_integer_forward_pass():
    model = ConvTranspose2dInteger(
        8, 4, 3, config={
            "weight_width": 8,
            "weight_frac_width": 4,
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    )
    model.bias = None
    input = torch.randn(1, 8, 16, 16)
    output = model(input)
    assert output.shape[1] == 4

def test_convtranspose2d_binary_forward_pass():
    model = ConvTranspose2dBinary(
        8, 4, 3, config={
            "bypass": True,
            "data_in_stochastic": False,
            "bias_stochastic": False,
            "weight_stochastic": False,
            "data_in_bipolar": False,
            "bias_bipolar": False,
            "weight_bipolar": False,
        }
    )
    input = torch.randn(1, 8, 16, 16)
    output = model(input)
    assert isinstance(output, torch.Tensor)

def test_convtranspose2d_minifloat_ieee_forward_pass():
    model = ConvTranspose2dMinifloatIEEE(
        8, 4, 3,
        config={
            "weight_width": 8,
            "weight_exponent_width": 3,
            "weight_exponent_bias": 4,
            "data_in_width": 8,
            "data_in_exponent_width": 3,
            "data_in_exponent_bias": 4,
            "bias_width": 8,
            "bias_exponent_width": 3,
            "bias_exponent_bias": 4,
        }
    )
    model.bias = None
    input = torch.randn(1, 8, 16, 16)
    output = model(input)
    assert output.shape[1] == 4
