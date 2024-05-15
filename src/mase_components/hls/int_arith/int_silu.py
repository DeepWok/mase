from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_silu_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
):
    """
    This script generates a fixed-point silu in HLS
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
            body += f"data.data_{i}_{j} = din.data_{i}_{j}/(1+hls::exp((ap_fixed<16, 8>)-din.data_{i}_{j}));\n"
    body += "data_out.write(data);} }"

    buff = f"""
// Relu:
void int_silu_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body}
}}
"""

    writer.code_buff += buff
    return writer
