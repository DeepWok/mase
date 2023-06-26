#! /usr/bin/env python3
# ---------------------------------------
# This script emits HLS code for the block float mac ops
# ---------------------------------------

import sys, os, time, math


def _clog2(x):
    return int(math.ceil(math.log2(x)))


def emit_block_float_cast(share_num=16, out_sig_width=3, in_sig_width=6, exp_width=8):
    """
    cast signifidants
    """
    buff = "void bfloat_cast("
    for i in range(0, share_num):
        buff += f"ap_int<{in_sig_width+1}> sig_x_{i},"
    buff += f"ap_uint<{exp_width}> exp_x,"
    for i in range(0, share_num):
        buff += f"ap_int<{out_sig_width+1}> *sig_y_{i},"
    buff += f"ap_uint<{exp_width}> *exp_y) {{"
    buff += "\n#pragma HLS INLINE\n"
    buff += "\n// TODO\n"
    for i in range(0, share_num):
        buff += f"*sig_y_{i} = sig_x_{i};"
    buff += f"*exp_y = exp_x;"

    buff += "}"
    return buff


def emit_block_float_mac(
    share_num=16,
    exp_width=8,
    sig_width=3,
):
    """
    compute a * b + c
    """

    header = """
// This file implements an 8-bit bfloat MAC operator.
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;

"""

    buff = "void bfloat_mac("
    for i in range(0, share_num):
        buff += f"ap_int<{sig_width+1}> sig_a_{i},"
    buff += f"ap_uint<{exp_width}> exp_a,"
    for i in range(0, share_num):
        buff += f"ap_int<{sig_width+1}> sig_b_{i},"
    buff += f"ap_uint<{exp_width}> exp_b,"
    for i in range(0, share_num):
        buff += f"ap_int<{sig_width+1}> sig_c_{i},"
    buff += f"ap_uint<{exp_width}> exp_c,"
    for i in range(0, share_num):
        buff += f"ap_int<{sig_width+1}> *sig_r_{i},"
    buff += f"ap_uint<{exp_width}> *exp_r) {{"
    buff += f"""
#pragma HLS PIPELINE II=1
ap_uint<{sig_width}> zero = 0;
"""

    # Multiplication
    buff += f"""
// ab = a * b
// The exponents are added
ap_uint<{exp_width+1}> exp_ab = exp_a + exp_b;
// The significands are multiplied 
"""
    for i in range(0, share_num):
        buff += f"ap_int<{2*sig_width}> sig_ab_{i} = sig_a_{i} * sig_b_{i};"

    exp_max = (1 << exp_width) - 1
    sig_max = (1 << sig_width) - 1

    # Addition
    buff += f"""

// r = ab + c
ap_int<{exp_width+2}> exp_ab_cast = exp_ab;
ap_int<{exp_width+2}> exp_c_cast = exp_c;
if (exp_c_cast > exp_ab_cast + 2) {{
// If c is too significant than ab, emit c directly.
  *exp_r = exp_c;
"""
    for i in range(0, share_num):
        buff += f"*sig_r_{i} = sig_c_{i};"
    buff += f"""}}

if (exp_ab_cast > exp_c_cast + 2) {{
// If ab is too significant than c, emit ab directly.
  if (exp_r[{exp_width}]) {{
// Saturated. Return inf/-inf
  *exp_r = {exp_max};
"""
    for i in range(0, share_num):
        buff += f"*sig_r_{i} = (sig_ab_{i}[{2*sig_width-1}], zero);"
    buff += f"""
}}
else
bfloat_cast(
"""
    for i in range(0, share_num):
        buff += f"sig_ab_{i},"
    buff += f"exp_ab,"
    for i in range(0, share_num):
        buff += f"sig_r_{i},"
    buff += f"exp_r);"
    buff += "}"
    header += emit_block_float_cast(
        share_num=share_num,
        out_sig_width=sig_width,
        in_sig_width=2 * sig_width,
        exp_width=exp_width,
    )

    buff += f"ap_int<{exp_width+2}> shift = exp_ab_cast - exp_c_cast;"
    for i in range(0, share_num):
        buff += f"ap_int<{exp_width+2}> r_{i} = sig_ab_{i} + (sig_c_{i} << shift);"
    buff += "bfloat_cast("
    for i in range(0, share_num):
        buff += f"r_{i},"
    buff += f"exp_ab,"
    for i in range(0, share_num):
        buff += f"sig_r_{i},"
    buff += f"exp_r);"
    buff += "}"

    cpp_name = "bfloat_mac.cpp"
    with open(cpp_name, "w", encoding="utf-8") as outf:
        outf.write(header + buff)
    os.system(f"clang-format -i {cpp_name}")

    tcl = """
open_project -reset bfloat_mac
set_top bfloat_mac
add_files { ./bfloat_mac.cpp }
open_solution -reset "solution1"
set_part {xcu250-figd2104-2L-e}
create_clock -period 4 -name default
config_bind -effort high
config_compile -pipeline_loops 1
csynth_design
export_design -flow syn -format ip_catalog
"""

    tcl_name = "bfloat_mac.tcl"
    with open(tcl_name, "w", encoding="utf-8") as outf:
        outf.write(tcl)


emit_block_float_mac(
    share_num=1,
    exp_width=4,
    sig_width=3,
)
