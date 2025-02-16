import random, os, pytest
import numpy as np
import cocotb
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner

import torch
from torch import nn

DATA_IN_0_PRECISION_1 = 8
DATA_OUT_0_PRECISION_1 = 8


@cocotb.test()
async def cocotb_test_fixed_softsign(dut):
    # Range of values
    min_value = -4
    max_value = 4

    # Number of equidistant values
    num_values = 161

    # Calculate the resolution
    resolution = (max_value - min_value) / (num_values - 1)

    # Convert the resolution into fixed-point format
    resolution_fixed_point = int(resolution * (2**DATA_IN_0_PRECISION_1))

    # Generate the equidistant values
    values = np.linspace(min_value, max_value, num_values)

    # Convert values to fixed-point format
    values_fixed_point = np.round(values * (2**DATA_IN_0_PRECISION_1)).astype(int)

    tensor_tanh = torch.Tensor(values)

    model = nn.Tanh()
    tanh_values = model(tensor_tanh)

    max_error = 0.02

    for i in range(87):
        # a = tanh_values[i]
        b = max_value * (2**DATA_IN_0_PRECISION_1) - resolution_fixed_point * (i - 1)

        a = [b / (2**DATA_IN_0_PRECISION_1)]
        tensor_tanh = torch.Tensor(a)
        c = model(tensor_tanh)

        tanh_value_numpy = c.detach().numpy()

        # Scale Tanh output to fixed-point range and convert to integers
        scaled_value = np.round(
            tanh_value_numpy * (2**DATA_OUT_0_PRECISION_1)
        ).astype(int)

        dut.data_in_0[0].value = b

        await Timer(10, units="ns")

        ##result = multiplier_sw(data_a, data_b)

        assert (dut.data_out_0[0].value - scaled_value) <= (
            max_error * (2 ** (DATA_OUT_0_PRECISION_1))
        ), "Randomised test failed "


@pytest.mark.dev
def test_fixed_softsign():
    mase_runner(
        trace=True,
        module_param_list=[
            {
                "DATA_IN_0_PRECISION_0": DATA_IN_0_PRECISION_1 * 2,
                "DATA_IN_0_PRECISION_1": DATA_IN_0_PRECISION_1,
                "DATA_IN_0_TENSOR_SIZE_DIM_0": 1,
                "DATA_IN_0_PARALLELISM_DIM_0": 1,
                "DATA_IN_0_PARALLELISM_DIM_1": 1,
                "DATA_OUT_0_PRECISION_0": DATA_OUT_0_PRECISION_1 * 2,
                "DATA_OUT_0_PRECISION_1": DATA_OUT_0_PRECISION_1,
                "DATA_OUT_0_TENSOR_SIZE_DIM_0": 1,
                "DATA_OUT_0_PARALLELISM_DIM_0": 1,
                "DATA_OUT_0_PARALLELISM_DIM_1": 1,
            },
        ],
    )


if __name__ == "__main__":
    test_fixed_softsign()

""" if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()]) """
