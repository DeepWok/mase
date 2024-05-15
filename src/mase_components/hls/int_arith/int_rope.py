import math

from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_rope_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
    w_width=16,
    w_frac_width=8,
):
    """
    This script generates a fixed-point rope in HLS
    """
    assert writer is not None
    assert x_width >= x_frac_width
    assert x_width > 0

    op_id = writer.op_id
    writer.op_id += 1

    y_width = x_width
    y_frac_width = x_frac_width
    y_row = x_row
    y_col = x_col
    y_row_depth = x_row_depth
    y_col_depth = x_col_depth

    type_in = get_fixed_ty(x_row, x_col, x_width, x_frac_width)
    if type_in not in writer.types:
        writer.type_buff += new_fixed_ty(x_row, x_col, x_width, x_frac_width)
        writer.types.append(type_in)

    type_w = get_fixed_ty(x_row, x_col, y_width, y_frac_width)
    if type_w not in writer.types:
        writer.type_buff += new_fixed_ty(x_row, x_col, w_width, w_frac_width)
        writer.types.append(type_w)

    type_out = get_fixed_ty(y_row, y_col, y_width, y_frac_width)
    if type_out not in writer.types:
        writer.type_buff += new_fixed_ty(y_row, y_col, y_width, y_frac_width)
        writer.types.append(type_out)

    body_dispatch = ""
    body_dispatch += f"""
ap_fixed<{y_width}, {y_width-y_frac_width}> col_buff[{x_row_depth*x_row}][{x_col}];
int index = 0;
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {int(math.ceil(x_row_depth*1.5))}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
if (i == 0)
    index = 0;
if (i < {x_row_depth}) {{
{type_in} data=data_in.read();
"""
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_dispatch += f"col_buff[{x_row}*i+{i}][{j}] = data.data_{i}_{j};"
    body_dispatch += f"""
}}
if (i >= {int(math.ceil(x_row_depth*0.5))}) {{
{type_in} data_1, data_2;
int offset = {int(math.ceil(x_row*x_row_depth*0.5))};
int offset_index = (index > offset) ? index - offset : index + offset;
"""
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_dispatch += f"data_1.data_{i}_{j} = col_buff[index++];"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_dispatch += f"data_2.data_{i}_{j} = col_buff[offset_index++];"
    body_dispatch += f"""
    data_out_1.write(data_1);
    data_out_2.write(data_2);
}}}}}}
"""

    body_prod = ""
    for j in range(0, x_col):
        body_prod += (
            f"ap_fixed<{y_width}, {y_width-y_frac_width}> a_{j}[{x_col_depth}];\n"
        )
        body_prod += (
            f"ap_fixed<{y_width}, {y_width-y_frac_width}> b_{j}[{x_col_depth}];\n"
        )
    body_prod += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} data_1=data_in_1.read();
{type_in} data_2=data_in_2.read();
{type_out} data_d;
"""
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_prod += f"data_d.data_{i}_{j} = data_1.data_{i}_{j} * a_{j}[j] + data_2.data_{i}_{j} * b_{j}[j];\n"
    body_prod += f"""
data_out.write(data_d);
}}}}
"""

    buff = f"""
// RoPE:
void int_rope_dispatch_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_in}> &data_out_1, hls::stream<{type_in}> &data_out_2) {{
#pragma HLS INLINE OFF
{body_dispatch}
}}

void int_rope_prod_{op_id}(hls::stream<{type_in}> &data_in_1, hls::stream<{type_in}> &data_in_2, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body_prod}
}}

void int_rope_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE
#pragma HLS DATAFLOW 
hls::stream<{type_in}> data_1;
hls::stream<{type_in}> data_2;
int_rope_dispatch_{op_id}(data_in, data_1, data_2);
int_rope_prod_{op_id}(data_1, data_2, data_out);
}}
"""

    writer.code_buff += buff
    return writer
