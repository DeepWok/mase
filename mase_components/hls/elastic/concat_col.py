from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_concat_col_gen(
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
    This script generates a concat_col in HLS
    """
    assert writer is not None
    assert x_width >= x_frac_width
    assert x_width > 0

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

    writer.code_buff += buff.format(args, body, op_id=op_id)
    return writer
