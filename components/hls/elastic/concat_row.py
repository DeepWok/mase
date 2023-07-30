from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_concat_row_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
    concat_num=3,
):
    """
    This script generates a concat_row in HLS
    """
    assert writer is not None
    assert x_width >= x_frac_width
    assert x_width > 0

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

    writer.code_buff += buff.format(args, body, op_id=op_id)
    return writer
