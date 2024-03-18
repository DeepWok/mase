import sys
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from chop.passes.graph.transforms.quantize.quantizers.integer import *
import pdb
from bitstring import BitArray
from functools import partial

def make_quantizer(data_width:int, f_width:int):
    return partial(
                integer_quantizer, width=data_width, frac_width=f_width
            )

FUNCTION_TABLE = {
    'silu' : nn.SiLU(),
    'elu': nn.ELU(),
    'sigmoid': nn.Sigmoid(),
    'logsigmoid': nn.LogSigmoid(),
    'softshrink': nn.Softshrink(),
    'exp': torch.exp,
}

def fxtodouble(data_width: int, f_width: int, fx_num: str):
    intstr, fracstr = fx_num[:data_width-f_width], fx_num[data_width-f_width:]
    intval = float(BitArray(bin=intstr).int)
    fracval  = float(BitArray(bin=fracstr).uint) / 2 ** (f_width)

    return intval + fracval

def doubletofx(data_width: int, f_width: int, num: float, type = "hex"):
    assert type == "bin" or type == "hex", "type can only be: 'hex' or 'bin'"
    intnum = int(num * 2**(f_width))
    intbits = BitArray(int=intnum, length=data_width)
    return str(intbits.bin) if type == 'bin' else str(intbits)

def generate_lookup(data_width: int, f_width: int, function : str, type = "hex"):
    f = FUNCTION_TABLE[function]
    lut = {'data_width': data_width,
           'f_width' : f_width,
           'func' : FUNCTION_TABLE[function]}
    # entries = 2 ** data_width
    minval = float(-2 ** (data_width-f_width-1))
    maxval = (2**(data_width-1) - 1) * 2**(-f_width)
    i = minval
    quanter = make_quantizer(data_width, f_width)
    count = 0
    iarr = []
    while i <= maxval:
        count +=1
        iarr.append(i)
        val = quanter(f(torch.tensor(i))) # entry in the lookup table
        lut[doubletofx(data_width=data_width, f_width=f_width, num=i, type=type)] = doubletofx(data_width=data_width, f_width=f_width, num=val.item(), type=type)
        i+= 2 ** -(f_width)
    return lut

def aligned_generate_lookup(in_data_width, in_f_width, data_width: int, f_width: int, function : str, type = "hex"):
    f = FUNCTION_TABLE[function]
    lut = {'data_width': data_width,
           'f_width' : f_width,
           'in_data_width': data_width,
           'in_f_width' : f_width,
           'func' : FUNCTION_TABLE[function]
           }
    # entries = 2 ** data_width
    minval = float(-2 ** (in_data_width-in_f_width-1))
    maxval = (2**(in_data_width-1) - 1) * 2**(-in_f_width)
    inp_quanter = make_quantizer(in_data_width, in_f_width)
    quanter = make_quantizer(data_width, f_width)
    count = 0
    iarr = []
    pi = float(0)
    while pi <= maxval:
        count +=1
        iarr.append(pi)
        val = quanter(f(torch.tensor(pi))) # entry in the lookup table
        lut[doubletofx(data_width=in_data_width, f_width=in_f_width, num=pi, type=type)] = doubletofx(data_width=data_width, f_width=f_width, num=val.item(), type=type)
        pi += 2 ** -(in_f_width)
    i = minval
    while i <= -1 * 2**-(in_f_width):
        count +=1
        iarr.append(i)
        val = quanter(f(torch.tensor(i))) # entry in the lookup table
        lut[doubletofx(data_width=in_data_width, f_width=in_f_width, num=i, type=type)] = doubletofx(data_width=data_width, f_width=f_width, num=val.item(), type=type)
        i+= 2 ** -(in_f_width)
    iarr = [(x * 2 **(in_f_width)) for x in iarr]
    # print(iarr)
    return lut


def testlookup(lut):
    d = lut['data_width']
    f = lut['f_width']
    func = lut['func']
    idwidth = lut['in_data_width']
    ifracwidth = lut['in_f_width']
    quanter = make_quantizer(d,f)
    for k, v in lut.items():
        if v == d or v==f or v == func or v == idwidth or v == ifracwidth:
            continue
        inp = fxtodouble(idwidth,ifracwidth,k)
        outactual = func(torch.tensor(inp))
        outactual = quanter(outactual).item()
        outlut = fxtodouble(d,f,v)
        failed = (abs(outactual - outlut) > 0.001)
        if failed:
            print("bin val", k)
            print("to double", inp)
            print("double from nn silu", outactual)
            print(f"double from lut {outlut}, bin from lut {v}")
            print("\n")

def lookup_to_file(in_data_width, in_f_width, data_width: int, f_width: int, function: str, file_path = None):
    dicto = aligned_generate_lookup(in_data_width=in_data_width, in_f_width=in_f_width, data_width=data_width, f_width=f_width, function=function, type="bin")
    dicto = {k: v for k, v in dicto.items() if k not in ['data_width', 'f_width', 'func', 'in_data_width', 'in_f_width']}  
    with open(file_path, "w") as file:
    # Write values to the file separated by spaces
        file.write('\n'.join(str(value) for value in dicto.values()))
        file.write('\n')

def lookup_to_sv_file(data_width: int, f_width: int, function: str, file_path = None):
    dicto = aligned_generate_lookup(data_width=data_width, f_width=f_width, function=function, type="bin")
    dicto = {k: v for k, v in dicto.items() if k not in ['data_width', 'f_width', 'func']}  
    # Format for bit sizing
    key_format = f"{data_width}'b{{}}"
    value_format = f"{data_width}'b{{}}"

    # Starting the module and case statement
    sv_code = f"module {function}_lut_{data_width}_{f_width}(input logic [{data_width-1}:0] data_in_0, output logic [{data_width-1}:0] data_out_0);\n"
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

def generate_mem(function_name, in_data_width, in_f_width, data_width, f_width):
    assert function_name in FUNCTION_TABLE, f"Function {function_name} not found in FUNCTION_TABLE"
    lookup_to_file(in_data_width, in_f_width, data_width, f_width, function_name, f'/home/aw23/mase/machop/mase_components/activations/rtl/{function_name}_IN{in_data_width}_{in_f_width}_OUT{data_width}_{f_width}_map.mem')

def generate_sv_lut(function_name, data_width, f_width, dir = None):
    assert function_name in FUNCTION_TABLE, f"Function {function_name} not found in FUNCTION_TABLE"
    if dir is None:
        lookup_to_sv_file(data_width, f_width, function_name, f'/workspace/machop/mase_components/activations/rtl/{function_name}_map.sv')
    else:
        lookup_to_sv_file(data_width, f_width, function_name, f'{dir}/{function_name}_map_{data_width}_{f_width}.sv')


if __name__ == "__main__":
    # generate_sv_lut("silu", 8, 4)
    dicto = aligned_generate_lookup(in_data_width=16, in_f_width=8, data_width=8, f_width=4, function='exp', type="bin")
    # dicto = {k: v for k, v in dicto.items() if k not in ['data_width', 'f_width', 'func', 'in_data_width', 'in_f_width']}  
    testlookup(dicto)