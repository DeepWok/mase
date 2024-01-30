from .utils import clog2, get_fixed_ty, new_fixed_ty


def buffer_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
):
    """
    This script generates a buffer in HLS
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

    type_out = get_fixed_ty(y_row, y_col, y_width, y_frac_width)
    if type_out not in writer.types:
        writer.type_buff += new_fixed_ty(y_row, y_col, y_width, y_frac_width)
        writer.types.append(type_out)

    buff = f"""
// Buffer:
void buffer_{op_id}(
hls::stream<{type_in}> &data_in, 
hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF

{type_in} buffer[{x_row_depth}][{x_col_depth}];
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
buffer[i][j]=data_in.read();
}}
}}
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
data_out.write(buffer[i][j]);
}}
}}
}}
"""

    writer.code_buff += buff
    return writer
