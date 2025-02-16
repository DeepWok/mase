from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_transpose_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
):
    """
    This script generates a fixed-point transpose in HLS
    """
    assert writer is not None
    assert x_width >= x_frac_width
    assert x_width > 0

    op_id = writer.op_id
    writer.op_id += 1

    y_row = x_col
    y_col = x_row
    y_row_depth = x_col_depth
    y_col_depth = x_row_depth
    y_width = x_width
    y_frac_width = x_frac_width

    type_in = get_fixed_ty(x_row, x_col, x_width, x_frac_width)
    if type_in not in writer.types:
        writer.type_buff += new_fixed_ty(x_row, x_col, x_width, x_frac_width)
        writer.types.append(type_in)

    type_out = get_fixed_ty(y_row, y_col, y_width, y_frac_width)
    if type_out not in writer.types:
        writer.type_buff += new_fixed_ty(y_row, y_col, y_width, y_frac_width)
        writer.types.append(type_out)

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
void int_transpose_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}
"""
    writer.code_buff += buff
    return writer
