from .bfp_block_multiplier import bfp_block_multiplier_gen
from .utils import clog2, get_bfp_ty, new_bfp_ty


def bfp_mult_gen(
    writer,
    x_exp_width=16,
    x_man_width=8,
    x_row=3,
    x_col=2,
    x_row_depth=3,
    x_col_depth=2,
    w_exp_width=16,
    w_man_width=8,
):
    """
    This script generates a bfp mult in HLS
    """
    assert writer is not None
    assert x_man_width > 0
    assert x_exp_width > 0

    writer = bfp_block_multiplier_gen(
        writer,
        x_exp_width=x_exp_width,
        x_man_width=x_man_width,
        x_row=x_row,
        x_col=x_col,
        w_exp_width=w_exp_width,
        w_man_width=w_man_width,
    )
    block_mult_id = writer.op_id - 1

    y_exp_width = max(x_exp_width, w_exp_width) + 1
    y_man_width = x_man_width * w_man_width
    y_row = x_row
    y_col = x_col
    y_row_depth = x_row_depth
    y_col_depth = x_col_depth

    type_in0 = get_bfp_ty(x_row, x_col, x_exp_width, x_man_width)
    if type_in0 not in writer.types:
        writer.type_buff += new_bfp_ty(x_row, x_col, x_exp_width, x_man_width)
        writer.types.append(type_in0)

    type_in1 = get_bfp_ty(x_row, x_col, w_exp_width, w_man_width)
    if type_in1 not in writer.types:
        writer.type_buff += new_bfp_ty(x_row, x_col, w_exp_width, w_man_width)
        writer.types.append(type_in1)

    type_out = get_bfp_ty(y_row, y_col, y_exp_width, y_man_width)
    if type_out not in writer.types:
        writer.type_buff += new_bfp_ty(y_row, y_col, y_exp_width, y_man_width)
        writer.types.append(type_out)

    op_id = writer.op_id
    buff = f"""
// BFP Add:
void bfp_mult_{op_id}(hls::stream<{type_in0}> &data_in_0, 
hls::stream<{type_in1}> &data_in_1, 
hls::stream<{type_out}> &data_out) {{
#pragma HLS INLINE OFF
for (int j = 0; j < {x_col_depth}; j++) {{
for (int i = 0; i < {x_row_depth}; i++) {{
#pragma HLS LOOP_FLATTEN
#pragma HLS LOOP_MERGE
#pragma HLS PIPELINE II=1
{type_in0} d0=data_in_0.read();
{type_in1} d1=data_in_1.read();
{type_out} data;
bfp_block_multiplier_{block_mult_id}(d0, d1, &data);
data_out.write(data); }}}}
}}
"""

    writer.code_buff += buff
    writer.op_id += 1

    return writer
