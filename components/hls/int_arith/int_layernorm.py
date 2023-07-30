from .utils import clog2, get_fixed_ty, new_fixed_ty


def int_layernorm_gen(
    writer,
    x_width=16,
    x_frac_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
):
    """
    This script generates a fixed-point layernorm in HLS
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

    op_id = writer.op_id
    writer.op_id += 1

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

    type_mean = get_fixed_ty(1, y_col, y_width, y_frac_width)
    if type_mean not in writer.types:
        writer.type_buff += new_fixed_ty(1, y_col, y_width, y_frac_width)
        writer.types.append(type_mean)

    body_mean = ""
    for i in range(0, x_col):
        body_mean += f"ap_fixed<{y_width}, {y_width-y_frac_width}> mean_{i};\n"
    body_mean += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in} data=data_in.read();
if (i == 0) {{
"""
    for i in range(0, x_col):
        body_mean += f"mean_{i} = 0;\n"
    body_mean += "}"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_mean += f"ap_fixed<{y_width}, {y_width-y_frac_width}> d_{i}_{j} = data.data_{i}_{j};\n"
    for j in range(0, x_col):
        body_mean += f"mean_{j} += ("
        for i in range(0, x_row):
            body_mean += f"d_{i}_{j} + "
        body_mean = body_mean[: body_mean.rfind("+")] + f") / {x_row*x_row_depth};"
    body_mean += f"{type_out} dout;"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_mean += f"dout.data_{i}_{j} = d_{i}_{j};\n"
    body_mean += f"data_out.write(dout); if (i == {x_row_depth-1}) {{"
    body_mean += f"{type_mean} mean;"
    for i in range(0, x_col):
        body_mean += f"mean.data_0_{i} = mean_{i};\n"
    body_mean += f"data_mean.write(mean);}}}}}}"

    body_var = f"{type_mean} mean;"
    for i in range(0, x_col):
        body_var += f"ap_fixed<{y_width}, {y_width-y_frac_width}> mean_{i};\n"
    for i in range(0, x_col):
        body_var += f"ap_fixed<{y_width}, {y_width-y_frac_width}> var_{i} = 0;\n"
    body_var += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_out} data=data_in.read();
if (i == 0) {{
mean = data_mean.read();
"""
    for i in range(0, x_col):
        body_var += f"mean_{i} = mean.data_0_{i};\n"
    for i in range(0, x_col):
        body_var += f"var_{i} = 0;\n"
    body_var += "}"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_var += f"ap_fixed<{y_width}, {y_width-y_frac_width}> d_{i}_{j} = data.data_{i}_{j};\n"
    for j in range(0, x_col):
        body_var += f"var_{j} += "
        for i in range(0, x_row):
            body_var += f"(d_{i}_{j} - mean_{j})*(d_{i}_{j} - mean_{j}) + "
        body_var = body_var[: body_var.rfind("+")] + f";"
    body_var += f"data_out.write(data); if (i == {x_row_depth-1}) {{"
    body_var += f"{type_mean} var;"
    for i in range(0, x_col):
        body_var += f"var.data_0_{i} = hls::sqrt((ap_fixed<16, 8>)var_{i});\n"
    body_var += f"data_mean_out.write(mean); data_var.write(var);}}}}}}"

    body_ln = f"{type_mean} mean;"
    for i in range(0, x_col):
        body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> mean_{i};\n"
    body_ln += f"{type_mean} var;"
    for i in range(0, x_col):
        body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> var_{i};\n"
    body_ln += "\n// Added random constants here\n"
    for i in range(0, x_col):
        body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> weight_{i} = 2;\n"
    for i in range(0, x_col):
        body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> bias_{i} = 3;\n"
    body_ln += f"""
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_out} data=data_in.read();
if (i == 0) {{
mean = data_mean.read();
var = data_var.read();
"""
    for i in range(0, x_col):
        body_ln += f"mean_{i} = mean.data_0_{i};\n"
        body_ln += f"var_{i} = var.data_0_{i};\n"
    body_ln += "}"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_ln += f"ap_fixed<{y_width}, {y_width-y_frac_width}> d_{i}_{j} = data.data_{i}_{j};\n"
    body_ln += f"{type_out} dout;"
    for i in range(0, x_row):
        for j in range(0, x_col):
            body_ln += f"dout.data_{i}_{j} = (d_{i}_{j} - mean_{j})/var_{j} * weight_{j} + bias_{j};\n"
    body_ln += f"data_out.write(dout);}}}}"

    buff = f"""
// Layernorm:
void int_layernorm_mean_{op_id}(hls::stream<{type_in}> &data_in, 
hls::stream<{type_out}> &data_out, hls::stream<{type_mean}> &data_mean) {{
#pragma HLS INLINE OFF
{body_mean}
}}

void int_layernorm_var_{op_id}(hls::stream<{type_out}> &data_in, 
hls::stream<{type_out}> &data_out, 
hls::stream<{type_mean}> &data_mean, 
hls::stream<{type_mean}> &data_var,
hls::stream<{type_mean}> &data_mean_out) {{
#pragma HLS INLINE OFF
{body_var}
}}


void int_layernorm_ln_{op_id}(hls::stream<{type_out}> &data_in, 
hls::stream<{type_out}> &data_out, 
hls::stream<{type_mean}> &data_mean, 
hls::stream<{type_mean}> &data_var) {{
#pragma HLS INLINE OFF
{body_ln}
}}


void int_layernorm_{op_id}(hls::stream<{type_in}> &data_in, hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE
#pragma HLS DATAFLOW 
hls::stream<{type_mean}> data_mean, data_mean_out, data_var; 
hls::stream<{type_out}> data_buff_0, data_buff_1;
int_layernorm_mean_{op_id}(data_in, data_buff_0, data_mean);
int_layernorm_var_{op_id}(data_buff_0, data_buff_1, data_mean, data_var, data_mean_out);
int_layernorm_ln_{op_id}(data_buff_1, data_out, data_mean_out, data_var);
}}
"""
    writer.code_buff += buff
    return writer
