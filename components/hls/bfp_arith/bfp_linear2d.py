from .bfp_multiplier_float import bfp_multiplier_float_gen
from .utils import clog2, get_bfp_ty, new_bfp_ty


def bfp_linear2d_gen(
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
    b_exp_width=16,
    b_man_width=8,
):
    """
    This script generates a fixed-point linear2d in HLS
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

    writer = bfp_multiplier_float_gen(
        writer,
        x_exp_width=x_exp_width,
        x_man_width=x_man_width,
        w_exp_width=w_exp_width,
        w_man_width=w_man_width,
    )
    mult_id = writer.op_id - 1

    if x_exp_width > 5 or w_exp_width > 5:
        # Use fp32
        ew = 8
        mw = 23
        fpt = "float"
        union_ty = """union
{
  unsigned int intval;
  float fpval;
}
"""
    else:
        # Use fp16
        ew = 5
        mw = 10
        fpt = "half"
        union_ty = """union
{
  unsigned int intval;
  half fpval;
}
"""

    y_exp_width = ew
    y_man_width = mw

    type_in = get_bfp_ty(x_row, x_col, x_exp_width, x_man_width)
    if type_in not in writer.types:
        writer.type_buff += new_bfp_ty(x_row, x_col, x_exp_width, x_man_width)
        writer.types.append(type_in)

    type_out = get_bfp_ty(y_row, y_col, y_exp_width, y_man_width)
    if type_out not in writer.types:
        writer.type_buff += new_bfp_ty(y_row, y_col, y_exp_width, y_man_width)
        writer.types.append(type_out)

    body = ""
    body += f"ap_int<{w_exp_width}> weight_exp [{w_col_depth}][{w_row_depth}];\n"
    for i in range(0, w_row):
        for j in range(0, w_col):
            body += f"ap_int<{w_man_width+1}> weight_man_{i}_{j} [{w_col_depth}][{w_row_depth}];\n"
    body += f"ap_uint<{x_exp_width}> data_in_exp;\n"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body += f"ap_int<{x_man_width+1}> data_in_man_{i}_{j};\n"
    for i in range(0, y_row):
        for k in range(0, y_col):
            body += f"{fpt} data_{i}_{k} [{y_row_depth}];\n"
    for i in range(0, y_row):
        body += f"{fpt} bias_{i}[{y_row_depth}];\n"
    body += f"""
for (int i = 0; i < {x_col_depth}; i++) {{
for (int j = 0; j < {x_row_depth}; j++) {{
for (int k = 0; k < {w_row_depth}; k++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
"""
    body += f"if (k == 0) {{ {type_in} d = data_in.read();"
    body += f"data_in_exp = d.exponent;"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body += f"data_in_man_{i}_{j} = d.data_{i}_{j};"
    body += "}"
    # Begin of the complicated part
    body += f"if (j != {x_row_depth-1}) {{"
    for i in range(0, y_row):
        for j in range(0, y_col):
            body += f"data_{i}_{j}[k] += "
            for k in range(0, x_row):
                body += f"bfp_multiplier_float_{mult_id}(data_in_exp, data_in_man_{k}_{j}, weight_exp[j][k], weight_man_{i}_{k}[j][k]) + "
            body = body[: body.rfind("+")] + ";"
    body += "} else {"
    # End of the complicated part
    body += f"{type_out} d;"
    for i in range(0, y_row):
        for k in range(0, y_col):
            if b_exp_width == 0:
                body += f"{fpt} d_data_{i}_{k} = data_{i}_{k}[k] + "
            else:
                body += f"{fpt} d_data_{i}_{k} = data_{i}_{k}[k]+ bias_{i}[k] + "
            for k in range(0, x_row):
                body += f"bfp_multiplier_float_{mult_id}(data_in_exp, data_in_man_{k}_{j}, weight_exp[j][k], weight_man_{i}_{k}[j][k]) + "
            body = body[: body.rfind("+")] + ";"
    for i in range(0, y_row):
        for j in range(0, y_col):
            body += f"data_{i}_{j}[k] = 0;"

    # Casting back to bfp
    body += f"ap_uint<{ew}> exp_max = 0;"
    for i in range(0, y_row):
        for j in range(0, y_col):
            body += f"""
ap_uint<1> d_sign_{i}_{j};
ap_uint<{ew}> d_exp_{i}_{j};
ap_uint<{mw}> d_man_{i}_{j};
ap_uint<{1+ew+mw}> val_{i}_{j};

{union_ty} u_{i}_{j}; 
u_{i}_{j}.fpval = d_data_{i}_{j};
val_{i}_{j} = u_{i}_{j}.intval;
(d_sign_{i}_{j}, d_exp_{i}_{j}, d_man_{i}_{j}) = val_{i}_{j};
exp_max = (exp_max > d_exp_{i}_{j}) ? exp_max :  d_exp_{i}_{j};
"""

    body += f"d.exponent = exp_max;"
    for i in range(0, y_row):
        for j in range(0, y_col):
            body += f"d.data_{i}_{j} = (d_sign_{i}_{j}, d_man_{i}_{j}) >> (exp_max - d_exp_{i}_{j});"
    body += "data_out.write(d);"
    body += "}}}}"

    linear_buff = """
// Linear 2D:
void bfp_linear2d_{}(hls::stream<{}> &data_in, hls::stream<{}> &data_out) {{
#pragma HLS INLINE OFF
{}
}}
""".format(
        writer.op_id, type_in, type_out, body
    )

    writer.code_buff += linear_buff
    writer.op_id += 1

    return writer
