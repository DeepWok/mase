from .utils import clog2, get_fixed_ty, new_fixed_ty


def fork_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
    fork_num=2,
):
    """
    This script generates a fork in HLS
    """
    assert writer is not None
    assert x_width >= x_frac_width
    assert x_width > 0

    op_id = writer.op_id
    writer.op_id += 1

    type_in = get_fixed_ty(x_row, x_col, x_width, x_frac_width)
    if type_in not in writer.types:
        writer.type_buff += new_fixed_ty(x_row, x_col, x_width, x_frac_width)
        writer.types.append(type_in)

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
void fork_{op_id}(hls::stream<{type_in}> &data_in, {args}) {{
#pragma HLS INLINE OFF
{body}
}}
"""

    writer.code_buff += buff
    return writer
