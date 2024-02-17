from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_linear2d_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
    w_width=16,
    w_frac_width=8,
    w_row=7,
    w_col=3,
    w_row_depth=4,
    w_col_depth=3,
    b_width=16,
    b_frac_width=8,
):
    """
    This script generates a fixed-point linear2d in HLS
    """
    assert writer is not None
    assert x_width >= x_frac_width
    assert w_width >= w_frac_width
    assert b_width >= b_frac_width
    assert x_width > 0
    assert w_width > 0

    assert w_col_depth == x_row_depth, f"{w_col_depth} != {x_row_depth}"
    assert w_col == x_row
    y_row = w_row
    y_col = x_col
    y_row_depth = w_row_depth
    y_col_depth = x_col_depth

    x_int_width = x_width - x_frac_width
    w_int_width = w_width - w_frac_width
    b_int_width = b_width - b_frac_width
    y_int_width = max(x_int_width + w_int_width, b_int_width)
    y_frac_width = max(x_frac_width + w_frac_width, b_frac_width)

    y_width = y_int_width + y_frac_width

    type_in = get_fixed_ty(x_row, x_col, x_width, x_frac_width)
    if type_in not in writer.types:
        writer.type_buff += new_fixed_ty(x_row, x_col, x_width, x_frac_width)
        writer.types.append(type_in)

    type_out = get_fixed_ty(y_row, y_col, y_width, y_frac_width)
    if type_out not in writer.types:
        writer.type_buff += new_fixed_ty(y_row, y_col, y_width, y_frac_width)
        writer.types.append(type_out)

    body = ""
    for i in range(0, w_row):
        for j in range(0, w_col):
            body += f"ap_fixed<{w_width}, {w_width-w_frac_width}> weight_{i}_{j} [{w_col_depth}][{w_row_depth}];\n"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body += f"ap_fixed<{x_width}, {x_width-x_frac_width}> data_in_{i}_{j};\n"
    for i in range(0, y_row):
        for k in range(0, y_col):
            body += f"ap_fixed<{y_width}, {y_width-y_frac_width}> data_{i}_{k} [{y_row_depth}];\n"
    for i in range(0, y_row):
        body += (
            f"ap_fixed<{y_width}, {y_width-y_frac_width}> bias_{i}[{y_row_depth}];\n"
        )
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
        for j in range(0, y_col):
            if b_width == 0:
                body += f"d.data_{i}_{j} = data_{i}_{j}[k] + "
            else:
                body += f"d.data_{i}_{j} = data_{i}_{j}[k]+ bias_{i}[k] + "
            for k in range(0, x_row):
                body += f"weight_{i}_{k}[j][k] * data_in_{k}_{j} + "
            body = body[: body.rfind("+")] + ";"
    for i in range(0, y_row):
        for j in range(0, y_col):
            body += f"data_{i}_{j}[k] = 0;"
    body += "data_out.write(d);"
    body += "}}}}"

    linear_buff = """
// Linear 2D:
void int_linear2d_{}(hls::stream<{}> &data_in, hls::stream<{}> &data_out) {{
#pragma HLS INLINE OFF
{}
}}
""".format(
        writer.op_id, type_in, type_out, body
    )

    writer.code_buff += linear_buff
    writer.op_id += 1

    return writer
