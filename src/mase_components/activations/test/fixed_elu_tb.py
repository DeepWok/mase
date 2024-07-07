#!/usr/bin/env python3

import pytest
import os, logging
import pdb
from bitstring import BitArray
import cocotb
from functools import partial
from cocotb.triggers import *
from chop.nn.quantizers.integer import *
import torch.nn.functional as F
import torch.nn as nn
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import (
    StreamDriver,
    StreamMonitor,
    StreamMonitorFloat,
)
from mase_cocotb.z_qlayers import quantize_to_int
from mase_cocotb.runner import mase_runner
from mase_cocotb.utils import bit_driver, sign_extend_t
from math import ceil
from pathlib import Path

# from chop.passes.graph.transforms.quantize.quantized_modules import LinearInteger

import torch

logger = logging.getLogger("testbench")
logger.setLevel(logging.INFO)


def make_quantizer(data_width: int, f_width: int):
    return partial(integer_quantizer, width=data_width, frac_width=f_width)


FUNCTION_TABLE = {
    "silu": nn.SiLU(),
    "elu": nn.ELU(),
    "sigmoid": nn.Sigmoid(),
    "logsigmoid": nn.LogSigmoid(),
    "softshrink": nn.Softshrink(),
    "gelu": nn.GELU(),
    "exp": torch.exp,
    "softmax": torch.exp,
}


def fxtodouble(data_width: int, f_width: int, fx_num: str):
    intstr, fracstr = fx_num[: data_width - f_width], fx_num[data_width - f_width :]
    intval = float(BitArray(bin=intstr).int)
    fracval = float(BitArray(bin=fracstr).uint) / 2 ** (f_width)

    return intval + fracval


def doubletofx(data_width: int, f_width: int, num: float, type="hex"):
    assert type == "bin" or type == "hex", "type can only be: 'hex' or 'bin'"
    intnum = int(num * 2 ** (f_width))
    intbits = BitArray(int=intnum, length=data_width)
    return str(intbits.bin) if type == "bin" else str(intbits)


def generate_lookup(data_width: int, f_width: int, function: str, type="hex"):
    f = FUNCTION_TABLE[function]
    lut = {
        "data_width": data_width,
        "f_width": f_width,
        "func": FUNCTION_TABLE[function],
    }
    # entries = 2 ** data_width
    minval = float(-(2 ** (data_width - f_width - 1)))
    maxval = (2 ** (data_width - 1) - 1) * 2 ** (-f_width)
    i = minval
    quanter = make_quantizer(data_width, f_width)
    count = 0
    iarr = []
    while i <= maxval:
        count += 1
        iarr.append(i)
        val = quanter(f(torch.tensor(i)))  # entry in the lookup table
        lut[doubletofx(data_width=data_width, f_width=f_width, num=i, type=type)] = (
            doubletofx(
                data_width=data_width, f_width=f_width, num=val.item(), type=type
            )
        )
        i += 2 ** -(f_width)
    return lut


def aligned_generate_lookup(
    in_data_width, in_f_width, data_width: int, f_width: int, function: str, type="hex"
):
    f = FUNCTION_TABLE[function]
    lut = {
        "data_width": data_width,
        "f_width": f_width,
        "in_data_width": data_width,
        "in_f_width": f_width,
        "func": FUNCTION_TABLE[function],
    }
    # entries = 2 ** data_width
    minval = float(-(2 ** (in_data_width - in_f_width - 1)))
    maxval = (2 ** (in_data_width - 1) - 1) * 2 ** (-in_f_width)
    inp_quanter = make_quantizer(in_data_width, in_f_width)
    quanter = make_quantizer(data_width, f_width)
    count = 0
    iarr = []
    pi = float(0)
    while pi <= maxval:
        count += 1
        iarr.append(pi)
        val = quanter(f(torch.tensor(pi)))  # entry in the lookup table
        lut[
            doubletofx(data_width=in_data_width, f_width=in_f_width, num=pi, type=type)
        ] = doubletofx(
            data_width=data_width, f_width=f_width, num=val.item(), type=type
        )
        pi += 2 ** -(in_f_width)

    i = minval
    while i <= -1 * 2 ** -(in_f_width):
        count += 1
        iarr.append(i)
        val = quanter(f(torch.tensor(i)))  # entry in the lookup table
        lut[
            doubletofx(data_width=in_data_width, f_width=in_f_width, num=i, type=type)
        ] = doubletofx(
            data_width=data_width, f_width=f_width, num=val.item(), type=type
        )
        i += 2 ** -(in_f_width)

    iarr = [(x * 2 ** (in_f_width)) for x in iarr]
    # print(iarr)
    return lut


def generate_elu(
    in_data_width, in_f_width, data_width: int, f_width: int, alpha=1.0, type="hex"
):
    f = nn.ELU(alpha)
    lut = {
        "data_width": data_width,
        "f_width": f_width,
        "in_data_width": data_width,
        "in_f_width": f_width,
        "func": f,
    }
    # entries = 2 ** data_width
    minval = float(-(2 ** (in_data_width - in_f_width - 1)))
    maxval = (2 ** (in_data_width - 1) - 1) * 2 ** (-in_f_width)
    inp_quanter = make_quantizer(in_data_width, in_f_width)
    quanter = make_quantizer(data_width, f_width)
    pi = float(0)
    while pi <= maxval:
        val = quanter(f(torch.tensor(pi)))  # entry in the lookup table
        lut[
            doubletofx(data_width=in_data_width, f_width=in_f_width, num=pi, type=type)
        ] = doubletofx(
            data_width=data_width, f_width=f_width, num=val.item(), type=type
        )
        pi += 2 ** -(in_f_width)
    i = minval
    while i <= -1 * 2 ** -(in_f_width):
        val = quanter(f(torch.tensor(i)))  # entry in the lookup table
        lut[
            doubletofx(data_width=in_data_width, f_width=in_f_width, num=i, type=type)
        ] = doubletofx(
            data_width=data_width, f_width=f_width, num=val.item(), type=type
        )
        i += 2 ** -(in_f_width)
    return lut


def checklookup(lut):
    d = lut["data_width"]
    f = lut["f_width"]
    func = lut["func"]
    idwidth = lut["in_data_width"]
    ifracwidth = lut["in_f_width"]
    quanter = make_quantizer(d, f)
    for k, v in lut.items():
        if v == d or v == f or v == func or v == idwidth or v == ifracwidth:
            continue
        inp = fxtodouble(idwidth, ifracwidth, k)
        outactual = func(torch.tensor(inp))
        outactual = quanter(outactual).item()
        outlut = fxtodouble(d, f, v)
        failed = abs(outactual - outlut) > 0.001
        if failed:
            print("bin val", k)
            print("to double", inp)
            print("double from nn silu", outactual)
            print(f"double from lut {outlut}, bin from lut {v}")
            print("\n")


def lookup_to_file(
    in_data_width,
    in_f_width,
    data_width: int,
    f_width: int,
    function: str,
    file_path=None,
):
    dicto = aligned_generate_lookup(
        in_data_width=in_data_width,
        in_f_width=in_f_width,
        data_width=data_width,
        f_width=f_width,
        function=function,
        type="bin",
    )
    dicto = {
        k: v
        for k, v in dicto.items()
        if k not in ["data_width", "f_width", "func", "in_data_width", "in_f_width"]
    }
    with open(file_path, "w") as file:
        # Write values to the file separated by spaces
        file.write("\n".join(str(value) for value in dicto.values()))
        file.write("\n")


def lookup_to_sv_file(
    in_data_width: int,
    in_f_width: int,
    data_width: int,
    f_width: int,
    function: str,
    file_path=None,
    path_with_dtype=False,
):
    dicto = aligned_generate_lookup(
        in_data_width=in_data_width,
        in_f_width=in_f_width,
        data_width=data_width,
        f_width=f_width,
        function=function,
        type="bin",
    )
    dicto = {
        k: v
        for k, v in dicto.items()
        if k not in ["data_width", "f_width", "func", "in_data_width", "in_f_width"]
    }
    # Format for bit sizing
    key_format = f"{in_data_width}'b{{}}"
    value_format = f"{data_width}'b{{}}"
    if path_with_dtype:
        end = f"_{data_width}_{f_width}"
    else:
        end = ""
    # Starting the module and case statement
    sv_code = f"""
`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
module {function}_lut{end} #(
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PRECISION_1 = 8
)
(
    /* verilator lint_off UNUSEDSIGNAL */
    input logic [{in_data_width-1}:0] data_in_0, 
    output logic [{data_width-1}:0] data_out_0
);
    
"""
    sv_code += "    always_comb begin\n"
    sv_code += "        case(data_in_0)\n"

    # Adding each case
    for key, value in dicto.items():
        formatted_key = key_format.format(key)
        formatted_value = value_format.format(value)
        sv_code += f"            {formatted_key}: data_out_0 = {formatted_value};\n"

    # Ending the case statement and module
    sv_code += f"            default: data_out_0 = {data_width}'b0;\n"
    sv_code += "        endcase\n"
    sv_code += "    end\n"
    sv_code += "endmodule\n"

    # Write the code to a SystemVerilog file
    with open(file_path, "w") as file:
        file.write(sv_code)

    print(f"SystemVerilog module generated and saved as {file_path}.")


def generate_sv_lut(
    function_name,
    in_data_width,
    in_f_width,
    data_width,
    f_width,
    dir=None,
    path_with_dtype=False,
):
    assert (
        function_name in FUNCTION_TABLE
    ), f"Function {function_name} not found in FUNCTION_TABLE"

    if path_with_dtype:
        end = f"_{data_width}_{f_width}"
    else:
        end = ""
    if dir is None:
        p = Path(__file__).parents[1] / "rtl"
        lookup_to_sv_file(
            in_data_width,
            in_f_width,
            data_width,
            f_width,
            function_name,
            str(p / f"{function_name}_lut{end}.sv"),
            path_with_dtype=path_with_dtype,
        )
    else:
        lookup_to_sv_file(
            in_data_width,
            in_f_width,
            data_width,
            f_width,
            function_name,
            f"{dir}/{function_name}_lut{end}.sv",
            path_with_dtype=path_with_dtype,
        )


def split_and_flatten_2d_tensor(input_tensor, row_block_size, col_block_size):
    rows, cols = input_tensor.size()

    num_row_blocks = rows // row_block_size
    num_col_blocks = cols // col_block_size

    reshaped_tensor = input_tensor.view(
        num_row_blocks, row_block_size, num_col_blocks, col_block_size
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()
    flattened_tensor = reshaped_tensor.view(-1, row_block_size * col_block_size)
    return flattened_tensor


class fixed_elu_tb(Testbench):
    def __init__(self, module, dut, dut_params, float_test=False) -> None:
        super().__init__(dut, dut.clk, dut.rst)

        self.data_width = dut_params["DATA_IN_0_PRECISION_0"]
        self.frac_width = dut_params["DATA_IN_0_PRECISION_1"]

        self.outputwidth = dut_params["DATA_OUT_0_PRECISION_0"]
        self.outputfracw = dut_params["DATA_OUT_0_PRECISION_1"]

        self.num_in_features = dut_params["DATA_IN_0_TENSOR_SIZE_DIM_0"]
        self.num_in_batches = dut_params["DATA_IN_0_TENSOR_SIZE_DIM_1"]

        self.size_in_feature_blocks = dut_params["DATA_IN_0_PARALLELISM_DIM_0"]
        self.size_in_batch_blocks = dut_params["DATA_IN_0_PARALLELISM_DIM_1"]

        self.num_in_feature_splits = int(
            ceil(self.num_in_features / self.size_in_feature_blocks)
        )
        self.num_in_batch_splits = int(
            ceil(self.num_in_batches / self.size_in_batch_blocks)
        )

        self.num_out_features = dut_params["DATA_OUT_0_TENSOR_SIZE_DIM_0"]
        self.num_out_batches = dut_params["DATA_OUT_0_TENSOR_SIZE_DIM_1"]

        self.size_out_feature_blocks = dut_params["DATA_OUT_0_PARALLELISM_DIM_0"]
        self.size_out_batch_blocks = dut_params["DATA_OUT_0_PARALLELISM_DIM_1"]

        self.num_out_feature_splits = int(
            ceil(self.num_out_features / self.size_out_feature_blocks)
        )
        self.num_out_batch_splits = int(
            ceil(self.num_out_batches / self.size_out_batch_blocks)
        )

        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        if float_test:
            self.data_out_0_monitor = StreamMonitorFloat(
                dut.clk,
                dut.data_out_0,
                dut.data_out_0_valid,
                dut.data_out_0_ready,
                self.outputwidth,
                self.outputfracw,
            )
        else:
            self.data_out_0_monitor = StreamMonitor(
                dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
            )

        self.in_dquantizer = partial(
            integer_quantizer,
            width=self.data_width,
            frac_width=self.frac_width,
            is_signed=True,
        )

        self.out_dquantizer = partial(
            integer_quantizer,
            width=self.outputwidth,
            frac_width=self.outputfracw,
            is_signed=True,
        )

        self.model = module
        self.real_in_tensor = torch.randn(self.num_in_batches, self.num_in_features)
        self.real_inp = self.real_in_tensor
        self.quant_in_tensor = self.in_dquantizer(self.real_in_tensor)
        self.real_out_tensor = self.model(self.quant_in_tensor)

        logger.info(f"REAL IN TENSOR: \n{self.real_in_tensor}")
        logger.info(f"REAL OUT TENSOR: \n{self.real_out_tensor}")

    # def exp(self):
    #     m = self.model(self.real_inp)
    #     m = split_and_flatten_2d_tensor(m, self.size_out_feature_blocks, self.size_out_batch_blocks)
    #     logger.info(f'IN EXP - FLOAT OUTPUT: \n{m}')
    #     m = self.out_dquantizer(m)
    #     logger.info(f'IN EXP - DQ OUTPUT: \n{m}')
    #     # mout = m.clamp(min=-1*2**(self.outputwidth-1), max = 2**(self.outputwidth-1)-1)
    #     print(f"Output width: {self.outputwidth}, output frac width: {self.outputfracw}")

    #     m2 = (m * 2 ** self.outputfracw).to(torch.int64)
    #     print(m2)
    #     m2 = torch.where(m2 < 0, (m2.clone().detach() % (2**self.outputwidth)), m2)
    #     return m2

    def exp(self):
        # Run the model with the provided inputs and return the expected integer outputs in the format expected by the monitor
        m = split_and_flatten_2d_tensor(
            self.real_out_tensor,
            self.size_out_batch_blocks,
            self.size_out_feature_blocks,
        )  # match output
        logger.info(f"EXP - FLOAT OUTPUT: \n{m}")
        m = self.out_dquantizer(m)
        m2 = (m * 2**self.outputfracw).to(torch.int64)
        m2 = m2.clone().detach() % (2**self.outputwidth)

        return m2

    def generate_inputs(self):
        # Generate the integer inputs for the DUT in the format expected by the driver
        inputs = split_and_flatten_2d_tensor(
            self.real_in_tensor, self.size_in_batch_blocks, self.size_in_feature_blocks
        )
        logger.info(f"FLOAT INPUT: \n{inputs}")
        inputs = self.in_dquantizer(inputs)
        intinp = (inputs * 2**self.frac_width).to(torch.int64)
        return intinp, inputs

    def doubletofx(self, num, data_width, f_width, type="bin"):
        assert type == "bin" or type == "hex", "type can only be: 'hex' or 'bin'"
        intnum = int(num * 2 ** (f_width))
        intbits = BitArray(int=intnum, length=data_width)
        return str(intbits.bin) if type == "bin" else str(intbits)

    async def run_test(self):
        await self.reset()
        logger.info(f"Reset finished")
        self.data_out_0_monitor.ready.value = 1
        for i in range(10):
            inputs, real_tensor = self.generate_inputs()
            exp_out = self.exp()
            logger.info(f"exp out {exp_out}")
            inputs = inputs.tolist()
            exp_out = exp_out.tolist()
            logger.info("Inputs and expected generated")
            logger.info(f"DUT IN: {inputs}")
            logger.info(f"DUT EXP OUT: {exp_out}")
            self.data_in_0_driver.load_driver(inputs)
            self.data_out_0_monitor.load_monitor(exp_out)

        await Timer(1000, units="us")
        assert self.data_out_0_monitor.exp_queue.empty()


@cocotb.test()
async def cocotb_test(dut):
    in_data_width = dut_params["DATA_IN_0_PRECISION_0"]
    in_frac_width = dut_params["DATA_IN_0_PRECISION_1"]
    out_data_width = dut_params["DATA_OUT_0_PRECISION_0"]
    out_frac_width = dut_params["DATA_OUT_0_PRECISION_1"]
    generate_sv_lut("elu", in_data_width, in_frac_width, out_data_width, out_frac_width)
    print("Generated memory")
    tb = fixed_elu_tb(torch.nn.ELU(), dut, dut_params, float_test=False)
    await tb.run_test()


dut_params = {
    "DATA_IN_0_TENSOR_SIZE_DIM_0": 12,
    "DATA_IN_0_TENSOR_SIZE_DIM_1": 12,
    "DATA_IN_0_PARALLELISM_DIM_0": 4,
    "DATA_IN_0_PARALLELISM_DIM_1": 3,
    "DATA_IN_0_PRECISION_0": 8,
    "DATA_IN_0_PRECISION_1": 4,
    "DATA_OUT_0_PRECISION_0": 8,
    "DATA_OUT_0_PRECISION_1": 4,
    "DATA_OUT_0_TENSOR_SIZE_DIM_0": 12,
    "DATA_OUT_0_TENSOR_SIZE_DIM_1": 12,
    "DATA_OUT_0_PARALLELISM_DIM_0": 4,
    "DATA_OUT_0_PARALLELISM_DIM_1": 3,
}

torch.manual_seed(1)


@pytest.mark.dev
def test_fixed_elu():
    # generate_memory.generate_sv_lut("exp", dut_params["DATA_IN_0_PRECISION_0"], dut_params["DATA_IN_0_PRECISION_1"])
    generate_sv_lut(
        "elu",
        dut_params["DATA_IN_0_PRECISION_0"],
        dut_params["DATA_IN_0_PRECISION_1"],
        dut_params["DATA_OUT_0_PRECISION_0"],
        dut_params["DATA_OUT_0_PRECISION_1"],
    )
    print("Generated memory")
    mase_runner(module_param_list=[dut_params])


if __name__ == "__main__":
    test_fixed_elu()
