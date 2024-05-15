from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_softmax_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
):
    """
    This script generates a fixed-point softmax in HLS
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

    type_expsum = get_fixed_ty(1, y_col, y_width, y_frac_width)
    if type_expsum not in writer.types:
        writer.type_buff += new_fixed_ty(1, y_col, y_width, y_frac_width)
        writer.types.append(type_expsum)

    body_exp = ""
    for i in range(0, x_col):
        body_exp += f"ap_fixed<{y_width}, {y_width-y_frac_width}> sum_{i};\n"
    body_exp += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} data=data_in.read();
if (i == 0) {{
"""
    for i in range(0, x_col):
        body_exp += f"sum_{i} = 0;\n"
    body_exp += "}"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_exp += f"ap_fixed<{y_width}, {y_width-y_frac_width}> d_{i}_{j} = hls::exp(data.data_{i}_{j});\n"
    for j in range(0, x_col):
        body_exp += f"sum_{j} += "
        for i in range(0, x_row):
            body_exp += f"d_{i}_{j} + "
        body_exp = body_exp[: body_exp.rfind("+")] + ";"
    body_exp += f"{type_out} dexp;"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_exp += f"dexp.data_{i}_{j} = d_{i}_{j};\n"
    body_exp += f"data_out.write(dexp); if (i == {x_row_depth-1}) {{"
    body_exp += f"{type_expsum} es;"
    for i in range(0, x_col):
        body_exp += f"es.data_0_{i} = sum_{i};\n"
    body_exp += f"data_expsum.write(es);}}}}}}"

    body_sm = ""
    for i in range(0, x_col):
        body_sm += f"ap_fixed<{y_width}, {y_width-y_frac_width}> sum_{i};\n"
    body_sm += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_out} data=data_in.read();
{type_expsum} exp_sum;
if (i == 0) {{
    exp_sum = data_expsum.read();
"""
    for i in range(0, x_col):
        body_sm += f"sum_{i} = exp_sum.data_0_{i};\n"
    body_sm += f"}}{type_out} dout;"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_sm += f"dout.data_{i}_{j} = data.data_{i}_{j}/sum_{j};\n"
    body_sm += "data_out.write(dout);}}"

    buff = f"""
// Softmax:
void int_softmax_expsum_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out, hls::stream<{type_expsum}> &data_expsum) {{
#pragma HLS INLINE OFF
{body_exp}
}}

void int_softmax_sm_{op_id}(hls::stream<{type_out}> &data_in, hls::stream<{type_expsum}> &data_expsum, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
{body_sm}
}}

void int_softmax_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE
#pragma HLS DATAFLOW 
hls::stream<{type_expsum}> data_expsum;
hls::stream<{type_out}> data_exp;
int_softmax_expsum_{op_id}(data_in, data_exp, data_expsum);
int_softmax_sm_{op_id}(data_exp, data_expsum, data_out);
}}
"""

    writer.code_buff += buff
    return writer
