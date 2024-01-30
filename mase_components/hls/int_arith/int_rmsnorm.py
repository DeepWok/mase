from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_rmsnorm_gen(
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
    This script generates a fixed-point rmsnorm in HLS
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

    body_mean = ""
    body_mean += f"""
ap_fixed<{y_width}, {y_width-y_frac_width}> mean = 0;
{type_in} buff[{x_row_depth}][{x_col_depth}];
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} data=data_in.read();
"""
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_mean += f"mean += data.data_{i}_{j} * data.data_{i}_{j}/{x_row * x_row_depth * x_col * x_col_depth};\n"
    body_mean += f"""
buff[i][j] = data;}}}}
ap_fixed<{y_width}, {y_width-y_frac_width}> mean_w = 1/mean;
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_out} data;
"""
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_mean += f"data.data_{i}_{j} = buff[i][j].data_{i}_{j}*mean_w;"
    body_mean += f"""
data_out.write(data);
}}}}
"""

    body_prod = ""
    for j in range(0, x_col):
        body_prod += (
            f"ap_fixed<{y_width}, {y_width-y_frac_width}> w_{j}[{x_col_depth}];\n"
        )
    body_prod += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_out} data=data_in.read();
{type_out} data_d;
"""
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_prod += f"data_d.data_{i}_{j} = data.data_{i}_{j} * w_{j}[j];\n"
    body_prod += f"""
data_out.write(data_d);
}}}}
"""

    buff = f"""
// RMSNorm:
void int_rmsnorm_mean_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body_mean}
}}

void int_rmsnorm_prod_{op_id}(hls::stream<{type_out}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body_prod}
}}

void int_rmsnorm_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE
#pragma HLS DATAFLOW 
hls::stream<{type_in}> data_buff;
int_rmsnorm_mean_{op_id}(data_in, data_buff);
int_rmsnorm_prod_{op_id}(data_buff, data_out);
}}
"""

    writer.code_buff += buff
    return writer
