import sys
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from chop.nn.quantizers.integer import *
import pdb
from bitstring import BitArray
from functools import partial
import mase_components
from pathlib import Path


def make_quantizer(data_width: int, f_width: int, floor):
    base_quantizer = integer_floor_quantizer if floor else integer_quantizer
    return partial(base_quantizer, width=data_width, frac_width=f_width)


def isqrt(x):
    x = (x + 1e-5).sqrt().reciprocal()
    return x


FUNCTION_TABLE = {
    "silu": nn.SiLU(),
    "elu": nn.ELU(),
    "sigmoid": nn.Sigmoid(),
    "logsigmoid": nn.LogSigmoid(),
    "softshrink": nn.Softshrink(),
    "gelu": nn.GELU(),
    "exp": torch.exp,
    "softmax": torch.exp,
    "isqrt": isqrt,
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

def inttobit(data_width:int, num: float, signed: bool = True):
    intbits = BitArray(int=num, length=data_width) if signed else BitArray(uint=num, length=data_width)
    return intbits

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
    in_data_width,
    in_f_width,
    data_width: int,
    f_width: int,
    function: str,
    type="hex",
    constant_mult=1,
    floor=False,
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
    inp_quanter = make_quantizer(in_data_width, in_f_width, floor)
    quanter = make_quantizer(data_width, f_width, floor)
    count = 0
    iarr = []
    pi = float(0)
    while pi <= maxval:
        count += 1
        iarr.append(pi)
        val = quanter(f(torch.tensor(pi * constant_mult)))  # entry in the lookup table
        lut[
            doubletofx(data_width=in_data_width, f_width=in_f_width, num=pi, type=type)
        ] = doubletofx(
            data_width=data_width, f_width=f_width, num=val.item(), type=type
        )
        pi += 2 ** -(in_f_width)

    if function not in ["isqrt"]:
        i = minval
        while i <= -1 * 2 ** -(in_f_width):
            count += 1
            iarr.append(i)
            val = quanter(
                f(torch.tensor(i * constant_mult))
            )  # entry in the lookup table
            lut[
                doubletofx(
                    data_width=in_data_width, f_width=in_f_width, num=i, type=type
                )
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


def testlookup(lut):
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
    constant_mult=1,
    floor=False,
):
    dicto = aligned_generate_lookup(
        in_data_width=in_data_width,
        in_f_width=in_f_width,
        data_width=data_width,
        f_width=f_width,
        function=function,
        type="bin",
        constant_mult=constant_mult,
        floor=floor,
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
    parameter DATA_IN_0_PRECISION_0  = 16,
    parameter DATA_IN_0_PRECISION_1  = 8,
    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PRECISION_1 = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [7:0] data_in_0,
    output logic [7:0] data_out_0
);

"""
    sv_code += """
  always_comb begin
    case (data_in_0)
"""

    # Adding each case
    for key, value in dicto.items():
        formatted_key = key_format.format(key)
        formatted_value = value_format.format(value)
        sv_code += f"      {formatted_key}: data_out_0 = {formatted_value};\n"

    # Ending the case statement and module
    sv_code += f"      default: data_out_0 = {data_width}'b0;\n"
    sv_code += "    endcase\n"
    sv_code += "  end\n"
    sv_code += "endmodule\n"

    # Write the code to a SystemVerilog file
    with open(file_path, "w") as file:
        file.write(sv_code)

    print(f"SystemVerilog module generated and saved as {file_path}.")


def inttobit(data_width:int, num: float, signed: bool = True):
    intbits = BitArray(int=num, length=data_width) if signed else BitArray(uint=num, length=data_width)
    return intbits
class GenerateSVLut:
    def __init__(self, function_name, parameter, path):
        assert (
            function_name in FUNCTION_TABLE
        ), f"Function {function_name} not found in FUNCTION_TABLE"
        self.f = FUNCTION_TABLE[function_name]
        self.parameter = parameter
        self.path = path
    def quant_profile(self, bin_in):
        bin_out = bin_in
        return bin_out

    def generate_lut_address(self):
        return NotImplementedError

    def generate_lut(self, lut_address: list):
        lut = {}
        for i in lut_address:
            bin_out = self.quant_profile(i)
            lut[i] = bin_out
        return lut

    def generate_sv(self,lut):
        self.generate_lut()
        return NotImplementedError

    def pipeline(self):
        lut_address = self.generate_lut_address(self)
        lut = self.generate_lut(lut_address)
        sv = self.generate_sv(lut)

from mase_components.linear_layers.mxint_operators.test.utils import mxint_quantize
class GenerateMxIntSVLut(GenerateSVLut):
    def quant_profile(self, bin_in):
        in_man_width, in_exp_width, out_man_width, out_exp_width = self.parameter["in_man_width"], self.parameter["in_exp_width"], self.parameter["out_man_width"], self.parameter["out_exp_width"]
        _bin = BitArray(bin=bin_in)
        exp_int = _bin[0:in_exp_width].int
        man_int = _bin[in_exp_width:in_man_width + in_exp_width].int
        value = man_int / 2**(in_man_width - 1) * 2**(exp_int)
        exp_value = self.f(torch.tensor(value))
        quant_value, mx, ex = mxint_quantize(exp_value,out_man_width,out_exp_width)
        exp_bit = inttobit(out_exp_width, num=ex).bin 
        man_bit = inttobit(out_man_width, num=mx).bin 
        bin_out = exp_bit + man_bit
        return bin_out
    def generate_lut_address(self):
        in_man_width, in_exp_width, out_man_width, out_exp_width = self.parameter["in_man_width"], self.parameter["in_exp_width"], self.parameter["out_man_width"], self.parameter["out_exp_width"]
        # we can determine the upperbound of exp
        from math import log
        upperbound_of_mx_output = (2**(out_man_width - 1) - 1) / 2**(out_man_width - 1) * 2**(2**(out_exp_width - 1) - 1)
        lowerbound_of_mx_output = (1) / 2**(out_man_width - 1) * 2**(-2**(out_exp_width - 1))
        positive_max_bound = log(upperbound_of_mx_output)
        negetive_max_bound = log(lowerbound_of_mx_output)
        # when input> max_bound or input < lower_boud, we actually dont need to represent them
        max_exp = torch.tensor(max(abs(positive_max_bound), abs(negetive_max_bound)))
        _, _, max_exp = mxint_quantize(max_exp)

        # actually, we also don't have that much precision to represent the data around 1(exp(0))
        # so the limitation at data around 0 can determine the minimum value of exp.
        # so we got two value in the left side or in the right side
        _left = (2**(out_man_width - 1) - 1) / 2**(out_man_width - 1)
        _right = (1*2**(out_man_width - 2) + 1) / 2**(out_man_width - 1) * 2**(1)
        # we need to find a way to rounding them, divide the gap by two, so when it's smaller than this value, we can actually think, it's 0 
        _left = 1 - (1 - _left)/2
        _right = 1 + (_right - 1)/2
        positive_min_bound = log(_left)
        negetive_min_bound = log(_right)
        min_exp = torch.tensor(min(abs(positive_min_bound), abs(negetive_min_bound)))
        _, _, min_exp = mxint_quantize(min_exp)
        address = []
        for i in range(int(min_exp), int(max_exp+in_man_width)):
            for j in range(2**in_man_width):
                exp_bin = inttobit(in_exp_width,i).bin
                man_bin = inttobit(in_man_width,j, signed=False).bin
                address += [str(exp_bin) + str(man_bin)]
        return address

def generate_sv_lut(
    function_name,
    in_data_width,
    in_f_width,
    data_width,
    f_width,
    path=None,  # maybe not accept path as a parameter due to redundantly-generated exp_lut
    path_with_dtype=False,
    constant_mult=1,
    floor=False,
):
    assert (
        function_name in FUNCTION_TABLE
    ), f"Function {function_name} not found in FUNCTION_TABLE"

    if path_with_dtype:
        end = f"_{data_width}_{f_width}"
    else:
        end = ""

    p = Path(__file__).parents[1] / "generated_lut" / "rtl"
    lookup_to_sv_file(
        in_data_width,
        in_f_width,
        data_width,
        f_width,
        function_name,
        str(p / f"{function_name}_lut{end}.sv"),
        path_with_dtype=path_with_dtype,
        constant_mult=constant_mult,
        floor=floor,
    )


if __name__ == "__main__":
    dwidths = [12]
    for func, _ in FUNCTION_TABLE.items():
        generate_sv_lut(func, 8, 4, data_width=8, f_width=4, path_with_dtype=False)

    # for k, v in FUNCTION_TABLE.items():
    # generate_sv_lut(k, 16, 8, 16, 8, dir="/home/bardia/code/adls/project/report_test", path_with_dtype=True)

    # dicto = aligned_generate_lookup(in_data_width=16, in_f_width=8, data_width=8, f_width=4, function='exp', type="bin")
    # # dicto = {k: v for k, v in dicto.items() if k not in ['data_width', 'f_width', 'func', 'in_data_width', 'in_f_width']}
    # testlookup(dicto)
