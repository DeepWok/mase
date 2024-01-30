# Temporary solution
from .utils import clog2, get_bfp_ty, new_bfp_ty


def bfp_mm_gen(
    writer,
    x_exp_width=16,
    x_man_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
    w_exp_width=16,
    w_man_width=8,
    w_row=7,
    w_col=3,
    w_row_depth=4,
    w_col_depth=3,
):
    """
    This script generates a fixed-point mm in HLS.
    Assume each column of x and each row of w is a block
    """
    assert writer is not None
    assert x_exp_width > 0
    assert w_exp_width > 0

    assert w_col_depth == x_row_depth, f"{w_col_depth} != {x_row_depth}"
    assert w_col == x_row
    y_row = w_row
    y_col = x_col
    y_row_depth = w_row_depth
    y_col_depth = x_col_depth

    bits = clog2(x_row)
    y_exp_width = max(x_exp_width, w_exp_width)
    y_man_width = max(x_man_width, w_man_width)

    min_shift = min(x_man_width, w_man_width)

    assert x_col == 1, "Current version only support x_col = 1"

    type_in = get_bfp_ty(x_row, x_col, x_exp_width, x_man_width)
    if type_in not in writer.types:
        writer.type_buff += new_bfp_ty(x_row, x_col, x_exp_width, x_man_width)
        writer.types.append(type_in)

    type_x = get_bfp_ty(x_row, 1, x_exp_width, x_man_width)
    if type_x not in writer.types:
        writer.type_buff += new_bfp_ty(x_row, 1, x_exp_width, x_man_width)
        writer.types.append(type_x)

    type_w = get_bfp_ty(1, w_col, w_exp_width, w_man_width)
    if type_w not in writer.types:
        writer.type_buff += new_bfp_ty(1, w_col, w_exp_width, w_man_width)
        writer.types.append(type_w)

    type_out = get_bfp_ty(y_row, y_col, y_exp_width, y_man_width)
    if type_out not in writer.types:
        writer.type_buff += new_bfp_ty(y_row, y_col, y_exp_width, y_man_width)
        writer.types.append(type_out)

    body = ""
    body += f"{type_in} d;\n"
    body += f"{type_out} dout_buff [{y_row_depth}];\n"
    body += f"""
for (int i = 0; i < {x_col_depth}; i++) {{
for (int j = 0; j < {x_row_depth}; j++) {{
for (int k = 0; k < {w_row_depth}; k++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
"""
    body += f"auto weight = weights.read(); if (k == 0) {{ d = data_in.read(); "
    # Begin of the complicated part
    body += f"if (j != {x_row_depth-1}) {{"
    for i in range(0, y_row):
        for j in range(0, y_col):
            # Single row of w and single column x - mult
            body += f"ap_uint<{max(x_exp_width, w_exp_width)+1}> wx_exp_{i}_{j} = d.exponent + weight.exponent;"
            body += f"ap_uint<{x_man_width+w_man_width+bits}> wx_man_{i}_{j} = "
            for k in range(0, x_row):
                body += f"weight.data_0_{k} * d.data_{k}_{j} + "
            body = body[: body.rfind("+")] + ";"
            for k in range(0, bits):
                body += f"if (wx_man_{i}_{j}[{x_man_width+w_man_width+bits-1-k}]) wx_exp_{i}_{j} += {bits-k};"
                if k != bits - 1:
                    body += " else "
            # TODO: Signess?
            body += f"ap_uint<{y_man_width}> wx_cast_man_{i}_{j} = wx_man_{i}_{j}.range({x_man_width+w_man_width-1}, {x_man_width+w_man_width-y_man_width});"

            true_body = ""
            false_body = ""
            for s in range(1, min_shift):
                true_body += f"""
if (diff == {s}) {{
dout_buff[k].data_{i}_{j} += (wx_cast_man_{i}_{j} >> {s});
}}
"""
                false_body += f"""
if (diff == {s}) {{
dout_buff[k].data_{i}_{j} = (dout_buff[k].data_{i}_{j}  >> {s}) + wx_cast_man_{i}_{j};
}}
"""

            body += f"""
{{
auto exp_0 = wx_exp_{i}_{j};
auto exp_1 = dout_buff[k].exponent;

if (exp_0 == exp_1) {{
dout_buff[k].data_{i}_{j} += wx_cast_man_{i}_{j};
}}
else if (exp_0 > exp_1 && exp_0 - exp_1 < {min_shift}) {{
ap_uint<{y_exp_width}> diff = exp_0 - exp_1;
{true_body}
}} 
else if (exp_1 > exp_0 && exp_1 - exp_0 < {min_shift}) {{
ap_uint<{y_exp_width}> diff = exp_1 - exp_0;
{false_body}
}}
}}
"""
    body += "} else {"
    # End of the complicated part
    body += f"{type_out} dout;"
    for i in range(0, y_row):
        for j in range(0, y_col):
            # Single row of w and single column x - mult
            body += f"ap_uint<{max(x_exp_width, w_exp_width)+1}> wx_exp_{i}_{j} = d.exponent + weight.exponent;"
            body += f"ap_uint<{x_man_width+w_man_width+bits}> wx_man_{i}_{j} = "
            for k in range(0, x_row):
                body += f"weight.data_0_{k} * d.data_{k}_{j} + "
            body = body[: body.rfind("+")] + ";"
            for k in range(0, bits):
                body += f"if (wx_man_{i}_{j}[{x_man_width+w_man_width+bits-1-k}]) wx_exp_{i}_{j} += {bits-k};"
                if k != bits - 1:
                    body += " else "
            # TODO: Signess?
            body += f"ap_uint<{y_man_width}> wx_cast_man_{i}_{j} = wx_man_{i}_{j}.range({x_man_width+w_man_width-1}, {x_man_width+w_man_width-y_man_width});"

            true_body = ""
            false_body = ""
            for s in range(1, min_shift):
                true_body += f"""
if (diff == {s}) {{
dout.data_{i}_{j} += dout_buff[k].data_{i}_{j}+ (wx_cast_man_{i}_{j} >> {s});
}}
"""
                false_body += f"""
if (diff == {s}) {{
dout.data_{i}_{j} = (dout_buff[k].data_{i}_{j} >> {s}) + wx_cast_man_{i}_{j};
}}
"""

            body += f"""
{{
auto exp_0 = wx_exp_{i}_{j};
auto exp_1 = dout_buff[k].exponent;

if (exp_0 == exp_1) {{
dout.data_{i}_{j} = dout_buff[k].data_{i}_{j} + wx_cast_man_{i}_{j};
}}
else if (exp_0 > exp_1 && exp_0 - exp_1 < {min_shift}) {{
ap_uint<{y_exp_width}> diff = exp_0 - exp_1;
{true_body}
}} 
else if (exp_1 > exp_0 && exp_1 - exp_0 < {min_shift}) {{
ap_uint<{y_exp_width}> diff = exp_1 - exp_0;
{false_body}
}}
}}
dout_buff[k].exponent = 0;
"""
    for i in range(0, y_row):
        for j in range(0, y_col):
            body += f"dout_buff[k].data_{i}_{j} = 0;"

    # TODO: Skipping normalization...
    body += "data_out.write(dout);"
    body += "}}}}}"

    linear_buff = """
// Linear 2D:
void bfp_mm_{}(hls::stream<{}> &data_in, 
hls::stream<{}> &weights,
hls::stream<{}> &data_out) {{
#pragma HLS INLINE OFF
{}
}}
""".format(
        writer.op_id, type_in, type_w, type_out, body
    )

    writer.code_buff += linear_buff
    writer.op_id += 1

    return writer
