#! /usr/bin/env python3
# ---------------------------------------
# This script emits HLS code for the transformer blocks
# ---------------------------------------

from argparse import ArgumentParser
import sys, os, time, logging, colorlog, glob, subprocess, multiprocessing, shutil, functools, math


def _clog2(x):
    return int(math.ceil(math.log2(x)))


def _get_fixed_ty(row, col, w, fw):
    return f"fixed_{row}_{col}_{w}_{fw}_t"


def _new_fixed_ty(row, col, w, fw):
    buff = f"struct {_get_fixed_ty(row, col, w, fw)} {{"
    for i in range(0, row):
        for j in range(0, col):
            buff += f"ap_fixed<{w}, {w-fw}> data_{i}_{j};"
    buff += "};"
    return buff


class TransformerHLSGenerator:
    def __init__(self, args):
        self.template = """
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
        self.type_buff = ""
        self.types = []
        self.op_id = 0

    def run(self):
        # buff = self.add_linear()
        # buff = self.add_block()
        # buff = self.add_attention()
        buff = self.add_opt_block_2()
        # buff = self.add_transpose()
        # buff = self.add_softmax()
        # buff = self.add_concat_col()
        # buff = self.add_concat_row()
        # buff = self.add_layernorm()
        # buff = self.add_relu()
        # buff = self.add_add()
        # buff = self.add_add_buff()
        # buff = self.add_fork()
        # buff = self.add_matrixmult()
        # buff = self.add_matrixmult_t()
        file_name = "./topx.cpp"
        tcl_name = "./vhlsx.tcl"
        final_buff = self.template + self.type_buff + buff
        with open(file_name, "w", encoding="utf-8") as outf:
            outf.write(final_buff)
        os.system(f"clang-format -i {file_name}")
        buff = f"""
open_project -reset top 
set_top opt_block_2_op_0
add_files {{ {file_name} }}
open_solution -reset "solution1"
set_part {{xcu250-figd2104-2L-e}}
create_clock -period 4 -name default
config_bind -effort high
config_compile -pipeline_loops 1
csynth_design
"""
        with open(tcl_name, "w", encoding="utf-8") as outf:
            outf.write(buff)

    def add_fork(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=4,
        x_col_depth=5,
        fork_num=2,
    ):
        op_id = self.op_id
        self.op_id += 1

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        args = ""
        for i in range(0, fork_num):
            args += f"hls::stream<{type_in}> &data_out_{i},"
        args = args[: args.rfind(",")]

        body = f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} din=data_in.read();
"""
        for i in range(0, fork_num):
            body += f"data_out_{i}.write(din);"
        body += "} }"

        buff = f"""
// Fork:
void fork_op_{op_id}(hls::stream<{type_in}> &data_in, {args}) {{
#pragma HLS INLINE OFF
{body}
}}
"""
        return buff

    def add_add_buff(
        self,
        x_0_width=8,
        x_0_frac_width=5,
        x_0_row=3,
        x_0_col=2,
        x_0_row_depth=4,
        x_0_col_depth=5,
        x_1_width=8,
        x_1_frac_width=5,
        x_1_row=3,
        x_1_col=2,
        x_1_row_depth=4,
        x_1_col_depth=5,
        y_width=8,
        y_frac_width=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        y_row = x_0_row
        y_col = x_0_col
        y_row_depth = x_0_row_depth
        y_col_depth = x_0_col_depth

        type_in0 = _get_fixed_ty(x_0_row, x_0_col, x_0_width, x_0_frac_width)
        if type_in0 not in self.types:
            self.type_buff += _new_fixed_ty(x_0_row, x_0_col, x_0_width, x_0_frac_width)
            self.types.append(type_in0)

        type_in1 = _get_fixed_ty(x_1_row, x_1_col, x_1_width, x_1_frac_width)
        if type_in1 not in self.types:
            self.type_buff += _new_fixed_ty(x_1_row, x_1_col, x_1_width, x_1_frac_width)
            self.types.append(type_in1)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        body = f"""
for (int j = 0; j < {x_0_col_depth}; j++) {{
for (int i = 0; i < {x_0_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in1} d1=data_in_1.read();
}}
}}
{type_in1} cache[{x_0_row_depth}][{x_0_col_depth}];
for (int j = 0; j < {x_0_col_depth}; j++) {{
for (int i = 0; i < {x_0_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in0} d0=data_in_0.read();
{type_in1} d1=cache[i][j];
{type_out} data;
"""
        for i in range(0, x_0_row):
            for j in range(0, x_0_col):
                body += f"data.data_{i}_{j} = d0.data_{i}_{j} + d1.data_{i}_{j};\n"
        body += "data_out.write(data);} }"

        buff = f"""
// Add:
void add_buff_op_{op_id}(
hls::stream<{type_in0}> &data_in_0, 
hls::stream<{type_in1}> &data_in_1, 
hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}
"""
        return buff

    def add_add(
        self,
        x_0_width=8,
        x_0_frac_width=5,
        x_1_width=8,
        x_1_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=4,
        x_col_depth=5,
        y_width=8,
        y_frac_width=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        y_row = x_row
        y_col = x_col
        y_row_depth = x_row_depth
        y_col_depth = x_col_depth

        type_in0 = _get_fixed_ty(x_row, x_col, x_0_width, x_0_frac_width)
        if type_in0 not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_0_width, x_0_frac_width)
            self.types.append(type_in0)

        type_in1 = _get_fixed_ty(x_row, x_col, x_1_width, x_1_frac_width)
        if type_in1 not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_1_width, x_1_frac_width)
            self.types.append(type_in1)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        body = f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in0} d0=data_in_0.read();
{type_in1} d1=data_in_1.read();
{type_out} data;
"""
        for i in range(0, x_row):
            for j in range(0, x_col):
                body += f"data.data_{i}_{j} = d0.data_{i}_{j} + d1.data_{i}_{j};\n"
        body += "data_out.write(data);} }"

        buff = f"""
// Add:
void add_op_{op_id}(hls::stream<{type_in0}> &data_in_0, 
hls::stream<{type_in1}> &data_in_1, 
hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}
"""
        return buff

    def add_relu(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=4,
        x_col_depth=5,
        y_width=8,
        y_frac_width=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        y_row = x_row
        y_col = x_col
        y_row_depth = x_row_depth
        y_col_depth = x_col_depth

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        body = f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} din=data_in.read();
{type_out} data;
"""
        for i in range(0, x_row):
            for j in range(0, x_col):
                body += f"if (din.data_{i}_{j}.is_neg()) data.data_{i}_{j} = 0; else data.data_{i}_{j} = din.data_{i}_{j};\n"
        body += "data_out.write(data);} }"

        buff = f"""
// Relu:
void relu_op_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}
"""
        return buff

    def add_layernorm(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=4,
        x_col_depth=5,
        y_width=8,
        y_frac_width=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        y_row = x_row
        y_col = x_col
        y_row_depth = x_row_depth
        y_col_depth = x_col_depth

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        type_mean = _get_fixed_ty(1, y_col, y_width, y_frac_width)
        if type_mean not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_mean)

        body_mean = ""
        for i in range(0, x_col):
            body_mean += f"ap_fixed<{y_width}, {y_width-y_frac_width}> mean_{i};\n"
        body_mean += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} data=data_in.read();
if (i == 0) {{
"""
        for i in range(0, x_col):
            body_mean += f"mean_{i} = 0;\n"
        body_mean += "}"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body_mean += f"ap_fixed<{y_width}, {y_width-y_frac_width}> d_{i}_{j} = data.data_{i}_{j};\n"
        for j in range(0, x_col):
            body_mean += f"mean_{j} += ("
            for i in range(0, x_row):
                body_mean += f"d_{i}_{j} + "
            body_mean = body_mean[: body_mean.rfind("+")] + f") / {x_row*x_row_depth};"
        body_mean += f"{type_out} dout;"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body_mean += f"dout.data_{i}_{j} = d_{i}_{j};\n"
        body_mean += f"data_out.write(dout); if (i == {x_row_depth-1}) {{"
        body_mean += f"{type_mean} mean;"
        for i in range(0, x_col):
            body_mean += f"mean.data_0_{i} = mean_{i};\n"
        body_mean += f"data_mean.write(mean);}}}}}}"

        body_var = f"{type_mean} mean;"
        for i in range(0, x_col):
            body_var += f"ap_fixed<{y_width}, {y_width-y_frac_width}> mean_{i};\n"
        for i in range(0, x_col):
            body_var += f"ap_fixed<{y_width}, {y_width-y_frac_width}> var_{i} = 0;\n"
        body_var += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_out} data=data_in.read();
if (i == 0) {{
mean = data_mean.read();
"""
        for i in range(0, x_col):
            body_var += f"mean_{i} = mean.data_0_{i};\n"
        for i in range(0, x_col):
            body_var += f"var_{i} = 0;\n"
        body_var += "}"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body_var += f"ap_fixed<{y_width}, {y_width-y_frac_width}> d_{i}_{j} = data.data_{i}_{j};\n"
        for j in range(0, x_col):
            body_var += f"var_{j} += "
            for i in range(0, x_row):
                body_var += f"(d_{i}_{j} - mean_{j})*(d_{i}_{j} - mean_{j}) + "
            body_var = body_var[: body_var.rfind("+")] + f";"
        body_var += f"data_out.write(data); if (i == {x_row_depth-1}) {{"
        body_var += f"{type_mean} var;"
        for i in range(0, x_col):
            body_var += f"var.data_0_{i} = hls::sqrt(var_{i});\n"
        body_var += f"data_mean_out.write(mean); data_var.write(var);}}}}}}"

        body_ln = f"{type_mean} mean;"
        for i in range(0, x_col):
            body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> mean_{i};\n"
        body_ln += f"{type_mean} var;"
        for i in range(0, x_col):
            body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> var_{i};\n"
        body_ln += "\n// Added random constants here\n"
        for i in range(0, x_col):
            body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> weight_{i} = 2;\n"
        for i in range(0, x_col):
            body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> bias_{i} = 3;\n"
        body_ln += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_out} data=data_in.read();
if (i == 0) {{
mean = data_mean.read();
var = data_var.read();
"""
        for i in range(0, x_col):
            body_ln += f"mean_{i} = mean.data_0_{i};\n"
            body_ln += f"var_{i} = var.data_0_{i};\n"
        body_ln += "}"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> d_{i}_{j} = data.data_{i}_{j};\n"
        body_ln += f"{type_out} dout;"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body_ln += f"dout.data_{i}_{j} = (d_{i}_{j} - mean_{j})/var_{j} * weight_{j} + bias_{j};\n"
        body_ln += f"data_out.write(dout);}}}}"

        buff = f"""
// Layernorm:
void layernorm_mean_{op_id}(hls::stream<{type_in}> &data_in, 
hls::stream<{type_out}> &data_out, hls::stream<{type_mean}> &data_mean) {{
#pragma HLS INLINE OFF
{body_mean}
}}

void layernorm_var_{op_id}(hls::stream<{type_out}> &data_in, 
hls::stream<{type_out}> &data_out, 
hls::stream<{type_mean}> &data_mean, 
hls::stream<{type_mean}> &data_var,
hls::stream<{type_mean}> &data_mean_out) {{
#pragma HLS INLINE OFF
{body_var}
}}


void layernorm_ln_{op_id}(hls::stream<{type_out}> &data_in, 
hls::stream<{type_out}> &data_out, 
hls::stream<{type_mean}> &data_mean, 
hls::stream<{type_mean}> &data_var) {{
#pragma HLS INLINE OFF
{body_ln}
}}


void layernorm_op_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE
#pragma HLS DATAFLOW 
hls::stream<{type_mean}> data_mean, data_mean_out, data_var; 
hls::stream<{type_out}> data_buff_0, data_buff_1;
hlayernorm_mean_{op_id}(data_in, data_buff_0, data_mean);
layernorm_var_{op_id}(data_buff_0, data_buff_1, data_mean, data_var, data_mean_out);
layernorm_ln_{op_id}(data_buff_1, data_out, data_mean_out, data_var);
}}
"""
        return buff

    def add_concat_row(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=4,
        x_col_depth=5,
        concat_num=3,
    ):
        op_id = self.op_id
        self.op_id += 1
        buff = """
// Concat_row:
void concat_row_op_{op_id}({}) {{
#pragma HLS INLINE OFF
{}
}}
"""
        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        args = ""
        for i in range(0, concat_num):
            args += f"hls::stream<{type_in}> &data_in_{i}, "
        args += f"hls::stream<{type_in}> &data_out"

        body = ""
        body += f"{type_in} buffer_0 [{x_row_depth*(x_col_depth-1)}];"
        for i in range(1, concat_num):
            body += f"{type_in} buffer_{i} [{x_row_depth*x_col_depth}];"

        body += f"""
ap_uint<{max(1, _clog2(x_col_depth*concat_num*x_row_depth))}> idx = 0;
for (int j = 0; j < {x_col_depth*x_row_depth}; j+={x_row_depth}) {{
for (int k = 0; k < {concat_num}; k++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} d, d0;
// Load data to buffer (the first column goes straight to output)
if (idx < {x_row_depth*x_col_depth}) {{
d0 = data_in_0.read();
"""
        for i in range(1, concat_num):
            body += f"buffer_{i}[idx] = data_in_{i}.read();"
        body += f"""
}}
if (idx >= {x_row_depth} &&idx < {x_row_depth*x_col_depth}) {{
buffer_0[idx-{x_row_depth}] = d0;
}}
// The first column goes straight to output
if (idx < {x_row_depth}) {{
d = d0;
}}
if (idx >= {x_row_depth} && k == 0) {{
d = buffer_0[j+i-4];
}}
"""
        for i in range(1, concat_num):
            body += f"""if (k == {i}) {{
d = buffer_{i}[j+i];
}}
"""
        body += "data_out.write(d);idx++;}}}"

        return buff.format(args, body, op_id=op_id)

    def add_concat_col(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=4,
        x_col_depth=5,
        concat_num=3,
    ):
        op_id = self.op_id
        self.op_id += 1
        buff = """
// Concat_col:
void concat_col_op_{op_id}({}) {{
#pragma HLS INLINE OFF
{}
}}
"""
        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        args = ""
        for i in range(0, concat_num):
            args += f"hls::stream<{type_in}> &data_in_{i}, "
        args += f"hls::stream<{type_in}> &data_out"

        body = ""
        for i in range(1, concat_num):
            body += f"{type_in} buffer_{i} [{x_row_depth}][{x_col_depth}];"

        body += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
data_out.write(data_in_0.read());
"""
        for i in range(1, concat_num):
            body += f"buffer_{i}[i][j] = data_in_{i}.read();"
        body += "}}"

        body += f"""
for (int k = 1; k < {concat_num}; k++) {{
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} d;
"""
        for i in range(1, concat_num):
            body += f"if (k == {i}) d = buffer_{i}[i][j];"
        body += "data_out.write(d);}}}"

        return buff.format(args, body, op_id=op_id)

    def add_softmax(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=4,
        x_col_depth=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        y_width = x_width
        y_frac_width = x_frac_width
        y_row = x_row
        y_col = x_col
        y_row_depth = x_row_depth
        y_col_depth = x_col_depth

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        type_expsum = _get_fixed_ty(1, y_col, y_width, y_frac_width)
        if type_expsum not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_expsum)

        body_exp = ""
        for i in range(0, x_col):
            body_exp += f"ap_fixed<{y_width}, {y_width-y_frac_width}> sum_{i};\n"
        body_exp += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} data=data_in.read();
if (i == 0) {{
"""
        for i in range(0, x_col):
            body_exp += f"sum_{i} = 0;\n"
        body_exp += "}"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body_exp += f"ap_fixed<{y_width}, {y_width-y_frac_width}> d_{i}_{j} = hls::exp(data.data_{i}_{j});\n"
        for j in range(0, x_col):
            body_exp += f"sum_{j} += "
            for i in range(0, x_row):
                body_exp += f"d_{i}_{j} + "
            body_exp = body_exp[: body_exp.rfind("+")] + ";"
        body_exp += f"{type_out} dexp;"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body_exp += f"dexp.data_{i}_{j} = d_{i}_{j};\n"
        body_exp += f"data_out.write(dexp); if (i == {x_row_depth-1}) {{"
        body_exp += f"{type_expsum} es;"
        for i in range(0, x_col):
            body_exp += f"es.data_0_{i} = sum_{i};\n"
        body_exp += f"data_expsum.write(es);}}}}}}"

        body_sm = ""
        for i in range(0, x_col):
            body_sm += f"ap_fixed<{y_width}, {y_width-y_frac_width}> sum_{i};\n"
        body_sm += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_out} data=data_in.read();
{type_expsum} exp_sum;
if (i == 0) {{
    exp_sum = data_expsum.read();
"""
        for i in range(0, x_col):
            body_sm += f"sum_{i} = exp_sum.data_0_{i};\n"
        body_sm += f"}}{type_out} dout;"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body_sm += f"dout.data_{i}_{j} = data.data_{i}_{j}/sum_{j};\n"
        body_sm += "data_out.write(dout);}}"

        buff = f"""
// Softmax:
void softmax_expsum_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out, hls::stream<{type_expsum}> &data_expsum) {{
#pragma HLS INLINE OFF
{body_exp}
}}

void softmax_sm_{op_id}(hls::stream<{type_out}> &data_in, hls::stream<{type_expsum}> &data_expsum, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body_sm}
}}

void softmax_op_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE
#pragma HLS DATAFLOW 
hls::stream<{type_expsum}> data_expsum;
hls::stream<{type_out}> data_exp;
softmax_expsum_{op_id}(data_in, data_exp, data_expsum);
softmax_sm_{op_id}(data_exp, data_expsum, data_out);
}}
"""

        return buff

    def add_transpose(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=4,
        x_col_depth=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        y_row = x_col
        y_col = x_row
        y_row_depth = x_col_depth
        y_col_depth = x_row_depth
        y_width = x_width
        y_frac_width = x_frac_width

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        body = f"""
{type_in} data[{x_row_depth}][{x_col_depth}];
for (int i = 0; i < {x_row_depth}; i++) {{
for (int j = 0; j < {x_col_depth}; j++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
data[i][j] = data_in.read();
}}
}}
for (int i = 0; i < {x_row_depth}; i++) {{
for (int j = 0; j < {x_col_depth}; j++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} din = data[j][i];
{type_out} dout;
"""
        for i in range(0, x_row):
            for j in range(0, y_row):
                body += f"dout.data_{j}_{i} = din.data_{i}_{j};"
        body += """
data_out.write(dout);
}
}
"""

        buff = f"""
// Transpose:
void transpose_op_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}
"""
        return buff

    def add_matrixmult_t(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=3,
        x_col_depth=2,
        w_width=8,
        w_frac_width=5,
        tw_row=3,
        tw_col=7,
        tw_row_depth=3,
        tw_col_depth=4,
        y_width=8,
        y_frac_width=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        # Transpose weights
        w_row_depth, w_col_depth = tw_col_depth, tw_row_depth
        w_row, w_col = tw_col, tw_row

        assert w_col_depth == x_row_depth
        assert w_col == x_row
        y_row = w_row
        y_col = x_col
        y_row_depth = w_row_depth
        y_col_depth = x_col_depth

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        type_win = _get_fixed_ty(tw_row, tw_col, w_width, w_frac_width)
        if type_win not in self.types:
            self.type_buff += _new_fixed_ty(tw_row, tw_col, w_width, w_frac_width)
            self.types.append(type_win)

        type_wout = _get_fixed_ty(w_row, w_col, w_width, w_frac_width)
        if type_wout not in self.types:
            self.type_buff += _new_fixed_ty(w_row, w_col, w_width, w_frac_width)
            self.types.append(type_wout)

        body_w = ""
        for i in range(0, w_row):
            for j in range(0, w_col):
                body_w += f"w.data_{i}_{j} = win.data_{j}_{i};\n"

        body = ""
        body += "\n// Start MM computation\n"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body += (
                    f"ap_fixed<{x_width}, {x_width-x_frac_width}> data_in_{i}_{j};\n"
                )
        for i in range(0, y_row):
            for k in range(0, y_col):
                body += f"ap_fixed<{y_width}, {y_width-y_frac_width}> data_{i}_{k} [{y_row_depth}];\n"
        body += f"""
for (int i = 0; i < {x_col_depth}; i++) {{
for (int j = 0; j < {x_row_depth}; j++) {{
for (int k = 0; k < {w_row_depth}; k++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_wout} weight = data_w.read();
"""
        body += f"if (k == 0) {{{type_in} d = data_in.read();"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body += f"data_in_{i}_{j} = d.data_{i}_{j};"
        body += "}"
        # Begin of the complicated part
        body += f"if (j != {x_row_depth-1}) {{"
        for i in range(0, y_row):
            for j in range(0, y_col):
                body += f"data_{i}_{j}[k] += "
                for k in range(0, x_row):
                    body += f"weight.data_{i}_{k} * data_in_{k}_{j} + "
                body = body[: body.rfind("+")] + ";"
        body += "} else {"
        # End of the complicated part
        body += f"{type_out} d;"
        for i in range(0, y_row):
            for k in range(0, y_col):
                body += f"d.data_{i}_{k} = data_{i}_{k}[k]+ "
                for k in range(0, x_row):
                    body += f"weight.data_{i}_{k} * data_in_{k}_{j} + "
                body = body[: body.rfind("+")] + ";"
        for i in range(0, y_row):
            for j in range(0, y_col):
                body += f"data_{i}_{j}[k] = 0;"
        body += "data_out.write(d);"
        body += "}}}}"

        matrixmult_t_buff = f"""
// Matrix mult_t:
// Cache the weights first
void matrixmult_t_weight_feed_{op_id}( 
hls::stream<{type_win}> &data_in, hls::stream<{type_wout}> &data_out) {{
#pragma HLS INLINE OFF
{type_wout} weights [{w_col_depth}][{w_row_depth}];\n
for (int i = 0; i < {w_col_depth}; i++) {{
for (int j = 0; j < {w_row_depth}; j++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_win} win = data_in.read();
{type_wout} w;
{body_w}
weights[i][j] = w;
}}
}}
for (int i = 0; i < {x_col_depth}; i++) {{
for (int j = 0; j < {x_row_depth}; j++) {{
for (int k = 0; k < {w_row_depth}; k++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
data_out.write(weights[j][k]);
}}
}}
}}
}}

// Compute MM 
void matrixmult_t_mm_{op_id}(hls::stream<{type_in}> &data_in, 
hls::stream<{type_wout}> &data_w, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}


// Compute MM Dataflow
void matrixmult_t_op_{op_id}(hls::stream<{type_in}> &data_in, 
hls::stream<{type_win}> &data_w, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE
#pragma HLS DATAFLOW 
hls::stream<{type_wout}> w;
  matrixmult_t_weight_feed_{op_id}(data_w, w);
  matrixmult_t_mm_{op_id}(data_in, w, data_out);
}}
"""

        return matrixmult_t_buff

    def add_matrixmult(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=3,
        x_col_depth=2,
        w_width=8,
        w_frac_width=5,
        w_row=7,
        w_col=3,
        w_row_depth=4,
        w_col_depth=3,
        y_width=8,
        y_frac_width=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        assert w_col_depth == x_row_depth
        assert w_col == x_row
        y_row = w_row
        y_col = x_col
        y_row_depth = w_row_depth
        y_col_depth = x_col_depth

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        type_w = _get_fixed_ty(w_row, w_col, w_width, w_frac_width)
        if type_w not in self.types:
            self.type_buff += _new_fixed_ty(w_row, w_col, w_width, w_frac_width)
            self.types.append(type_w)

        body = ""
        body += "\n// Start MM computation\n"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body += (
                    f"ap_fixed<{x_width}, {x_width-x_frac_width}> data_in_{i}_{j};\n"
                )
        for i in range(0, y_row):
            for k in range(0, y_col):
                body += f"ap_fixed<{y_width}, {y_width-y_frac_width}> data_{i}_{k} [{y_row_depth}];\n"
        body += f"""
for (int i = 0; i < {x_col_depth}; i++) {{
for (int j = 0; j < {x_row_depth}; j++) {{
for (int k = 0; k < {w_row_depth}; k++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_w} weight = data_w.read();
"""
        body += f"if (k == 0) {{{type_in} d = data_in.read();"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body += f"data_in_{i}_{j} = d.data_{i}_{j};"
        body += "}"
        # Begin of the complicated part
        body += f"if (j != {x_row_depth-1}) {{"
        for i in range(0, y_row):
            for j in range(0, y_col):
                body += f"data_{i}_{j}[k] += "
                for k in range(0, x_row):
                    body += f"weight.data_{i}_{k} * data_in_{k}_{j} + "
                body = body[: body.rfind("+")] + ";"
        body += "} else {"
        # End of the complicated part
        body += f"{type_out} d;"
        for i in range(0, y_row):
            for k in range(0, y_col):
                body += f"d.data_{i}_{k} = data_{i}_{k}[k]+ "
                for k in range(0, x_row):
                    body += f"weight.data_{i}_{k} * data_in_{k}_{j} + "
                body = body[: body.rfind("+")] + ";"
        for i in range(0, y_row):
            for j in range(0, y_col):
                body += f"data_{i}_{j}[k] = 0;"
        body += "data_out.write(d);"
        body += "}}}}"

        matrixmult_buff = f"""
// Matrix mult:
// Cache the weights first
void matrixmult_weight_feed_{op_id}( 
hls::stream<{type_w}> &data_in, hls::stream<{type_w}> &data_out) {{
#pragma HLS INLINE OFF
{type_w} weights [{w_col_depth}][{w_row_depth}];\n
for (int i = 0; i < {w_col_depth}; i++) {{
for (int j = 0; j < {w_row_depth}; j++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
weights[i][j] = data_in.read();
}}
}}
for (int i = 0; i < {x_col_depth}; i++) {{
for (int j = 0; j < {x_row_depth}; j++) {{
for (int k = 0; k < {w_row_depth}; k++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
data_out.write(weights[j][k]);
}}
}}
}}
}}

// Compute MM 
void matrixmult_mm_{op_id}(hls::stream<{type_in}> &data_in, 
hls::stream<{type_w}> &data_w, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}


// Compute MM Dataflow
void matrixmult_op_{op_id}(hls::stream<{type_in}> &data_in, 
hls::stream<{type_w}> &data_w, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE
#pragma HLS DATAFLOW 
hls::stream<{type_w}> w;
  matrixmult_weight_feed_{op_id}(data_w, w);
  matrixmult_mm_{op_id}(data_in, w, data_out);
}}
"""
        return matrixmult_buff

    def add_linear(
        self,
        x_width=8,
        x_frac_width=5,
        x_row=3,
        x_col=2,
        x_row_depth=3,
        x_col_depth=2,
        w_width=8,
        w_frac_width=5,
        w_row=7,
        w_col=3,
        w_row_depth=4,
        w_col_depth=3,
        y_width=8,
        y_frac_width=5,
    ):
        op_id = self.op_id
        self.op_id += 1

        assert w_col_depth == x_row_depth, f"{w_col_depth} != {x_row_depth}"
        assert w_col == x_row
        y_row = w_row
        y_col = x_col
        y_row_depth = w_row_depth
        y_col_depth = x_col_depth

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        body = ""
        for i in range(0, w_row):
            for j in range(0, w_col):
                body += f"ap_fixed<{w_width}, {w_width-w_frac_width}> weight_{i}_{j} [{w_col_depth}][{w_row_depth}];\n"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body += (
                    f"ap_fixed<{x_width}, {x_width-x_frac_width}> data_in_{i}_{j};\n"
                )
        for i in range(0, y_row):
            for k in range(0, y_col):
                body += f"ap_fixed<{y_width}, {y_width-y_frac_width}> data_{i}_{k} [{y_row_depth}];\n"
        for i in range(0, y_row):
            body += f"ap_fixed<{y_width}, {y_width-y_frac_width}> bias_{i}[{y_row_depth}];\n"
        body += f"""
for (int i = 0; i < {x_col_depth}; i++) {{
for (int j = 0; j < {x_row_depth}; j++) {{
for (int k = 0; k < {w_row_depth}; k++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
"""
        body += f"if (k == 0) {{ {type_in} d = data_in.read();"
        for i in range(0, x_row):
            for j in range(0, x_col):
                body += f"data_in_{i}_{j} = d.data_{i}_{j};"
        body += "}"
        # Begin of the complicated part
        body += f"if (j != {x_row_depth-1}) {{"
        for i in range(0, y_row):
            for j in range(0, y_col):
                body += f"data_{i}_{j}[k] += "
                for k in range(0, x_row):
                    body += f"weight_{i}_{k}[j][k] * data_in_{k}_{j} + "
                body = body[: body.rfind("+")] + ";"
        body += "} else {"
        # End of the complicated part
        body += f"{type_out} d;"
        for i in range(0, y_row):
            for k in range(0, y_col):
                body += f"d.data_{i}_{k} = data_{i}_{k}[k]+ bias_{i}[k] + "
                for k in range(0, x_row):
                    body += f"weight_{i}_{k}[j][k] * data_in_{k}_{j} + "
                body = body[: body.rfind("+")] + ";"
        for i in range(0, y_row):
            for j in range(0, y_col):
                body += f"data_{i}_{j}[k] = 0;"
        body += "data_out.write(d);"
        body += "}}}}"

        linear_buff = f"""
// Linear 2D:
void linear_op_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}
"""
        return linear_buff

    def add_attention(
        self,
        target_len=128,
        num_head=12,
        embed_dim=768,
        # num_head=16,
        # embed_dim=1024,
        # num_head=32,
        # embed_dim=2048,
        batch_size=1,
        x_row=32,
        x_col=1,
        x_width=8,
        x_frac_width=5,
        y_width=8,
        y_frac_width=5,
        w_qkv_row_depth=64,  # 1,
    ):
        op_id = self.op_id
        self.op_id += 1

        head_dim = embed_dim / num_head
        x_row_dim = embed_dim
        x_col_dim = target_len
        w_qkv_row_dim = head_dim
        w_qkv_col_dim = embed_dim

        assert x_row_dim % x_row == 0
        assert x_col_dim % x_col == 0

        x_row_depth = int(x_row_dim / x_row)
        x_col_depth = int(x_col_dim / x_col)
        w_qkv_col = x_row
        assert w_qkv_row_dim % w_qkv_row_depth == 0
        w_qkv_row = int(w_qkv_row_dim / w_qkv_row_depth)
        w_qkv_col_depth = x_row_depth

        final_buff = ""

        # -------------------------------------------------------
        #   Add types
        # -------------------------------------------------------

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        qkv_row = w_qkv_row
        qkv_col = x_col
        qkv_row_depth = w_qkv_row_depth
        qkv_col_depth = x_col_depth
        qkv_width = y_width
        qkv_frac_width = y_frac_width
        type_qkv = _get_fixed_ty(qkv_row, qkv_col, qkv_width, qkv_frac_width)
        if type_qkv not in self.types:
            self.type_buff += _new_fixed_ty(qkv_row, qkv_col, qkv_width, qkv_frac_width)
            self.types.append(type_qkv)

        a_row = qkv_col
        a_col = qkv_col
        a_row_depth = qkv_col_depth
        a_col_depth = qkv_col_depth
        a_width = y_width
        a_frac_width = y_frac_width
        type_a = _get_fixed_ty(a_row, a_col, a_width, a_frac_width)
        if type_a not in self.types:
            self.type_buff += _new_fixed_ty(a_row, a_col, a_width, a_frac_width)
            self.types.append(type_a)

        y_row = qkv_row
        y_col = a_col
        y_row_depth = qkv_row_depth
        y_col_depth = a_col_depth
        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        attention_buff = f"""
// Attention
void attention_op_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS DATAFLOW 
#pragma HLS INLINE
// Fork the input for QKV linear ops
hls::stream<{type_in}> xq, xk, xv;
fork_op_{op_id+1}(data_in, xq, xk, xv);

// QKV Linear
hls::stream<{type_qkv}> q, k, v;
linear_op_{op_id+2}(xq, q);
linear_op_{op_id+2}(xk, k);
linear_op_{op_id+2}(xv, v);

// A = QK^T
hls::stream<{type_a}> a, av;
hls::stream<{type_qkv}> bv0, bv1;
matrixmult_t_op_{op_id+3}(q, k, a);
fork_op_{op_id+4}(v, bv0);

// A^bar = softmax(A)
softmax_op_{op_id+5}(a, av);
fork_op_{op_id+4}(bv0, bv1);

// B = AV
matrixmult_op_{op_id+6}(av, bv1, data_out);
}}
"""

        # -------------------------------------------------------
        #   Add submodules - the following has to be added in a specific order,
        #   otherwise the op indices will mismatch
        # -------------------------------------------------------

        final_buff += self.add_fork(
            x_width=x_width,
            x_frac_width=x_frac_width,
            x_row=x_row,
            x_col=x_col,
            x_row_depth=x_row_depth,
            x_col_depth=x_col_depth,
            fork_num=3,
        )

        final_buff += self.add_linear(
            x_width=x_width,
            x_frac_width=x_frac_width,
            x_row=x_row,
            x_col=x_col,
            x_row_depth=x_row_depth,
            x_col_depth=x_col_depth,
            w_row=w_qkv_row,
            w_col=w_qkv_col,
            w_row_depth=w_qkv_row_depth,
            w_col_depth=w_qkv_col_depth,
            y_width=y_width,
            y_frac_width=y_frac_width,
        )

        final_buff += self.add_matrixmult_t(
            x_width=y_width,
            x_frac_width=y_frac_width,
            x_row=qkv_row,
            x_col=qkv_col,
            x_row_depth=qkv_row_depth,
            x_col_depth=qkv_col_depth,
            tw_row=qkv_row,
            tw_col=qkv_col,
            tw_row_depth=qkv_row_depth,
            tw_col_depth=qkv_col_depth,
            y_width=y_width,
            y_frac_width=y_frac_width,
        )

        final_buff += self.add_fork(
            x_width=y_width,
            x_frac_width=y_frac_width,
            x_row=qkv_row,
            x_col=qkv_col,
            x_row_depth=qkv_row_depth,
            x_col_depth=qkv_col_depth,
            fork_num=1,
        )

        final_buff += self.add_softmax(
            x_width=y_width,
            x_frac_width=y_frac_width,
            x_row=a_row,
            x_col=a_col,
            x_row_depth=a_row_depth,
            x_col_depth=a_col_depth,
        )

        final_buff += self.add_matrixmult(
            x_width=x_width,
            x_frac_width=x_frac_width,
            x_row=a_row,
            x_col=a_col,
            x_row_depth=a_row_depth,
            x_col_depth=a_col_depth,
            w_row=qkv_row,
            w_col=qkv_col,
            w_row_depth=qkv_row_depth,
            w_col_depth=qkv_col_depth,
            y_width=y_width,
            y_frac_width=y_frac_width,
        )

        return final_buff + attention_buff

    def add_opt_block_2(
        self,
        target_len=128,
        num_head=12,
        embed_dim=768,
        # num_head=16,
        # embed_dim=1024,
        # num_head=32,
        # embed_dim=2048,
        batch_size=1,
        x_row=32,
        x_col=1,
        x_width=8,
        x_frac_width=5,
        y_width=8,
        y_frac_width=5,
        w_qkv_row_depth=1,  # 64,  # 1,
        w_0_row_depth=768,  # 1,
        w_1_row_depth=4 * 768,  # 1,
        w_2_row_depth=24,  # 768,  # 1,
    ):
        op_id = self.op_id
        self.op_id += 1

        head_dim = embed_dim / num_head
        x_row_dim = embed_dim
        x_col_dim = target_len
        w_qkv_row_dim = head_dim
        w_qkv_col_dim = embed_dim

        assert x_row_dim % x_row == 0
        assert x_col_dim % x_col == 0

        x_row_depth = int(x_row_dim / x_row)
        x_col_depth = int(x_col_dim / x_col)
        w_qkv_col = x_row
        assert w_qkv_row_dim % w_qkv_row_depth == 0
        w_qkv_row = int(w_qkv_row_dim / w_qkv_row_depth)
        w_qkv_col_depth = x_row_depth

        w_0_row_dim = embed_dim
        w_0_col_dim = embed_dim
        w_1_row_dim = 4 * embed_dim
        w_1_col_dim = embed_dim
        w_2_row_dim = embed_dim
        w_2_col_dim = 4 * embed_dim
        assert w_0_row_dim % w_0_row_depth == 0
        w_0_row = int(w_0_row_dim / w_0_row_depth)
        w_0_col = w_qkv_row
        assert w_0_col_dim % w_0_col == 0
        w_0_col_depth = int(w_0_col_dim / w_0_col)
        assert w_1_row_dim % w_1_row_depth == 0
        w_1_row = int(w_1_row_dim / w_1_row_depth)
        w_1_col = w_0_row
        assert w_1_col_dim % w_1_col == 0
        w_1_col_depth = int(w_1_col_dim / w_1_col)
        assert w_2_row_dim % w_2_row_depth == 0
        w_2_row = int(w_2_row_dim / w_2_row_depth)
        w_2_col = w_1_row
        assert w_2_col_dim % w_2_col == 0
        w_2_col_depth = int(w_2_col_dim / w_2_col)

        # -------------------------------------------------------
        #   Add types
        # -------------------------------------------------------

        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        b_row = w_qkv_row
        b_col = x_col
        b_row_depth = w_qkv_row_depth * num_head
        b_col_depth = x_col_depth
        b_width = y_width
        b_frac_width = y_frac_width
        type_b = _get_fixed_ty(b_row, b_col, b_width, b_frac_width)
        if type_b not in self.types:
            self.type_buff += _new_fixed_ty(b_row, b_col, b_width, b_frac_width)
            self.types.append(type_b)

        bm_row = w_0_row
        bm_col = b_col
        bm_row_depth = w_0_row_depth
        bm_col_depth = b_col_depth
        bm_width = y_width
        bm_frac_width = y_frac_width
        type_bm = _get_fixed_ty(bm_row, bm_col, bm_width, bm_frac_width)
        if type_bm not in self.types:
            self.type_buff += _new_fixed_ty(bm_row, bm_col, bm_width, bm_frac_width)
            self.types.append(type_bm)

        b1_row = w_1_row
        b1_col = x_col
        b1_row_depth = w_1_row_depth
        b1_col_depth = x_col_depth
        b1_width = y_width
        b1_frac_width = y_frac_width
        type_b1 = _get_fixed_ty(b1_row, b1_col, b1_width, b1_frac_width)
        if type_b1 not in self.types:
            self.type_buff += _new_fixed_ty(b1_row, b1_col, b1_width, b1_frac_width)
            self.types.append(type_b1)

        y_row = w_2_row
        y_col = x_col
        y_row_depth = w_2_row_depth
        y_col_depth = x_col_depth
        assert y_row == x_row, f"y_row != x_row, {y_row} != {x_row}"
        assert y_col == x_col
        assert y_row_depth == x_row_depth
        assert y_col_depth == x_col_depth
        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        # -------------------------------------------------------
        #   Function body
        # -------------------------------------------------------

        opt_block_2_buff = f"""
// Attention
void opt_block_2_op_{op_id}( 
hls::stream<{type_in}> &x, 
hls::stream<{type_b}> &b,
hls::stream<{type_out}> &data_out) {{
#pragma HLS DATAFLOW 
#pragma HLS INLINE

// B_m = BW_0 + b+0
hls::stream<{type_bm}> bm, add, ln;
hls::stream<{type_in}> xb; 
linear_op_{op_id+1}(b, bm);
fork_op_{op_id+2}(x, xb);

// Add
add_buff_op_{op_id+3}(bm, xb, add);

// Layernorm
layernorm_op_{op_id+4}(add, ln);
hls::stream<{type_bm}> ln0, ln1, ln2, ln3, ln4;
fork_op_{op_id+5}(ln, ln0, ln1);

// Linear 1
hls::stream<{type_b1}> b1, relu;
linear_op_{op_id+6}(ln0, b1);
fork_op_{op_id+7}(ln1, ln2);

// ReLU
relu_op_{op_id+8}(b1, relu);
fork_op_{op_id+7}(ln2, ln3);

// Linear 1
hls::stream<{type_out}> b2, add2;
linear_op_{op_id+9}(relu, b2);
fork_op_{op_id+7}(ln3, ln4);

// Add
add_buff_op_{op_id+10}(b2, ln4, add2);

// Layernorm
layernorm_op_{op_id+11}(add2, data_out);

}}
"""
        # -------------------------------------------------------
        #   Add submodules - the following has to be added in a specific order,
        #   otherwise the op indices will mismatch
        # -------------------------------------------------------

        final_buff = ""

        final_buff += self.add_linear(
            x_width=b_width,
            x_frac_width=b_frac_width,
            x_row=b_row,
            x_col=b_col,
            x_row_depth=b_row_depth,
            x_col_depth=b_col_depth,
            w_row=w_0_row,
            w_col=w_0_col,
            w_row_depth=w_0_row_depth,
            w_col_depth=w_0_col_depth,
            y_width=y_width,
            y_frac_width=y_frac_width,
        )

        final_buff += self.add_fork(
            x_width=x_width,
            x_frac_width=x_frac_width,
            x_row=x_row,
            x_col=x_col,
            x_row_depth=x_row_depth,
            x_col_depth=x_col_depth,
            fork_num=1,
        )

        final_buff += self.add_add_buff(
            x_0_width=bm_width,
            x_0_frac_width=bm_frac_width,
            x_0_row=bm_row,
            x_0_col=bm_col,
            x_0_row_depth=bm_row_depth,
            x_0_col_depth=bm_col_depth,
            x_1_width=x_width,
            x_1_frac_width=x_frac_width,
            x_1_row=x_row,
            x_1_col=x_col,
            x_1_row_depth=x_row_depth,
            x_1_col_depth=x_col_depth,
            y_width=y_width,
            y_frac_width=y_frac_width,
        )

        final_buff += self.add_layernorm(
            x_width=bm_width,
            x_frac_width=bm_frac_width,
            x_row=bm_row,
            x_col=bm_col,
            x_row_depth=bm_row_depth,
            x_col_depth=bm_col_depth,
        )

        final_buff += self.add_fork(
            x_width=bm_width,
            x_frac_width=bm_frac_width,
            x_row=bm_row,
            x_col=bm_col,
            x_row_depth=bm_row_depth,
            x_col_depth=bm_col_depth,
            fork_num=2,
        )

        final_buff += self.add_linear(
            x_width=bm_width,
            x_frac_width=bm_frac_width,
            x_row=bm_row,
            x_col=bm_col,
            x_row_depth=bm_row_depth,
            x_col_depth=bm_col_depth,
            w_row=w_1_row,
            w_col=w_1_col,
            w_row_depth=w_1_row_depth,
            w_col_depth=w_1_col_depth,
            y_width=y_width,
            y_frac_width=y_frac_width,
        )

        final_buff += self.add_fork(
            x_width=bm_width,
            x_frac_width=bm_frac_width,
            x_row=bm_row,
            x_col=bm_col,
            x_row_depth=bm_row_depth,
            x_col_depth=bm_col_depth,
            fork_num=1,
        )

        final_buff += self.add_relu(
            x_width=b1_width,
            x_frac_width=b1_frac_width,
            x_row=b1_row,
            x_col=b1_col,
            x_row_depth=b1_row_depth,
            x_col_depth=b1_col_depth,
        )

        final_buff += self.add_linear(
            x_width=b1_width,
            x_frac_width=b1_frac_width,
            x_row=b1_row,
            x_col=b1_col,
            x_row_depth=b1_row_depth,
            x_col_depth=b1_col_depth,
            w_row=w_2_row,
            w_col=w_2_col,
            w_row_depth=w_2_row_depth,
            w_col_depth=w_2_col_depth,
            y_width=y_width,
            y_frac_width=y_frac_width,
        )

        final_buff += self.add_add_buff(
            x_0_width=y_width,
            x_0_frac_width=y_frac_width,
            x_0_row=y_row,
            x_0_col=y_col,
            x_0_row_depth=y_row_depth,
            x_0_col_depth=y_col_depth,
            x_1_width=bm_width,
            x_1_frac_width=bm_frac_width,
            x_1_row=bm_row,
            x_1_col=bm_col,
            x_1_row_depth=bm_row_depth,
            x_1_col_depth=bm_col_depth,
            y_width=y_width,
            y_frac_width=y_frac_width,
        )

        final_buff += self.add_layernorm(
            x_width=y_width,
            x_frac_width=y_frac_width,
            x_row=y_row,
            x_col=y_col,
            x_row_depth=y_row_depth,
            x_col_depth=y_col_depth,
        )

        return final_buff + opt_block_2_buff

    def add_block(
        self,
        target_len=128,
        num_head=12,
        embed_dim=768,
        # num_head=16,
        # embed_dim=1024,
        # num_head=32,
        # embed_dim=2048,
        batch_size=1,
        x_row=32,
        x_col=1,
        x_width=8,
        x_frac_width=5,
        y_width=8,
        y_frac_width=5,
        w_qkv_row_depth=64,  # 1,
        w_0_row_depth=768,  # 1,
        w_1_row_depth=4 * 768,  # 1,
        w_2_row_depth=768,  # 1,
    ):
        op_id = self.op_id
        self.op_id += 1

        head_dim = embed_dim / num_head
        x_row_dim = embed_dim
        x_col_dim = target_len
        w_qkv_row_dim = head_dim
        w_qkv_col_dim = embed_dim

        assert x_row_dim % x_row == 0
        assert x_col_dim % x_col == 0

        x_row_depth = int(x_row_dim / x_row)
        x_col_depth = int(x_col_dim / x_col)
        w_qkv_col = x_row
        assert w_qkv_row_dim % w_qkv_row_depth == 0
        w_qkv_row = int(w_qkv_row_dim / w_qkv_row_depth)
        w_qkv_col_depth = x_row_depth

        # -------------------------------------------------------
        #   Add types
        # -------------------------------------------------------

        # Attention part
        type_in = _get_fixed_ty(x_row, x_col, x_width, x_frac_width)
        if type_in not in self.types:
            self.type_buff += _new_fixed_ty(x_row, x_col, x_width, x_frac_width)
            self.types.append(type_in)

        b_row = w_qkv_row
        b_col = x_col
        b_row_depth = w_qkv_row_depth
        b_col_depth = x_col_depth
        type_b = _get_fixed_ty(b_row, b_col, b_width, b_frac_width)
        if type_b not in self.types:
            self.type_buff += _new_fixed_ty(b_row, b_col, b_width, b_frac_width)
            self.types.append(type_b)

        bm_row = w_0_row
        bm_col = b_col
        bm_row_depth = w_0_row_depth
        bm_col_depth = b_col_depth * num_head
        assert bm_row == x_row
        assert bm_col == x_col
        assert bm_row_depth == x_row_depth
        assert bm_col_depth == x_col_depth
        type_bm = _get_fixed_ty(bm_row, bm_col, bm_width, bm_frac_width)
        if type_bm not in self.types:
            self.type_buff += _new_fixed_ty(bm_row, bm_col, bm_width, bm_frac_width)
            self.types.append(type_bm)

        b1_row = w_1_row
        b1_col = b_col
        b1_row_depth = w_1_row_depth
        b1_col_depth = b_col_depth * num_head
        type_b1 = _get_fixed_ty(b1_row, b1_col, b1_width, b1_frac_width)
        if type_b1 not in self.types:
            self.type_buff += _new_fixed_ty(b1_row, b1_col, b1_width, b1_frac_width)
            self.types.append(type_b1)

        y_row = w_1_row
        y_col = b_col
        y_row_depth = w_1_row_depth
        y_col_depth = b_col_depth * num_head
        assert y_row == x_row
        assert y_col == x_col
        assert y_row_depth == x_row_depth
        assert y_col_depth == x_col_depth
        type_out = _get_fixed_ty(y_row, y_col, y_width, y_frac_width)
        if type_out not in self.types:
            self.type_buff += _new_fixed_ty(y_row, y_col, y_width, y_frac_width)
            self.types.append(type_out)

        # -------------------------------------------------------
        #   Function body
        # -------------------------------------------------------

        block_buff = f"""
// Attention
void attention_op_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS DATAFLOW 
#pragma HLS INLINE
// Fork the input for QKV linear ops
hls::stream<{type_in}> xq, xk, xv;
fork_op_{op_id+1}(data_in, xq, xk, xv);

// QKV Linear
hls::stream<{type_qkv}> q, k, v;
linear_op_{op_id+2}(xq, q);
linear_op_{op_id+2}(xk, k);
linear_op_{op_id+2}(xv, v);

// A = QK^T
hls::stream<{type_a}> a, av;
hls::stream<{type_qkv}> bv0, bv1;
matrixmult_t_op_{op_id+3}(q, k, a);
fork_op_{op_id+4}(v, bv0);

// A^bar = softmax(A)
softmax_op_{op_id+5}(a, av);
fork_op_{op_id+4}(bv0, bv1);

// B = AV
matrixmult_op_{op_id+6}(av, bv1, data_out);
}}
"""

        # !TODO: Need to check
        x_row = w_row
        x_col = x_col
        x_row_depth = w_row_depth
        x_col_depth = x_col_depth

        final_buff += self.add_concat_row(
            x_width=x_width,
            x_frac_width=x_frac_width,
            x_row=x_row,
            x_col=x_col,
            x_row_depth=x_row_depth,
            x_col_depth=x_col_depth,
            concat_num=num_head,
        )

        body += f"concat_row_out_{next_id}_t bm; concat_row_op_{next_id}("
        for i in range(0, num_head):
            body += f"b_{i},"
        body += f"bm);fork_op_{buf_id}(buffer_x_4, buffer_x_5);"
        next_id += 1

        # -------------------------------------------------------
        #   (fork) + Layernorm 1
        # -------------------------------------------------------

        # -------------------------------------------------------
        #   Linear -> ReLu -> Linear
        # -------------------------------------------------------

        # -------------------------------------------------------
        #   Layernorm 2
        # -------------------------------------------------------

        # return final_buff + block_buff
        return block_buff


# ---------- main function --------------
def main():
    parser = ArgumentParser()

    args = parser.parse_args()
    tg = TransformerHLSGenerator(args)
    tg.run()


if __name__ == "__main__":
    main()
