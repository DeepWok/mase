from .bfp_adder import bfp_adder_gen
from .utils import clog2, get_bfp_ty, new_bfp_ty


def bfp_block_adder_gen(
    writer,
    x_exp_width=16,
    x_man_width=8,
    x_row=3,
    x_col=2,
    w_exp_width=16,
    w_man_width=8,
):
    """
    This script generates a block-level bfp add in HLS
    """
    assert writer is not None
    assert x_man_width > 0
    assert x_exp_width > 0

    if x_exp_width > 5 or w_exp_width > 5:
        # Use fp32
        ew = 8
        mw = 23
    else:
        # Use fp16
        ew = 5
        mw = 10

    adder_id = writer.op_id
    writer = bfp_adder_gen(writer, exp_width=ew, man_width=mw)

    y_frac_width = max(x_man_width, w_man_width)
    y_width = (
        max(x_exp_width - x_man_width, w_exp_width - w_man_width) + 1 + y_frac_width
    )
    y_row = x_row
    y_col = x_col

    type_in0 = get_bfp_ty(x_row, x_col, x_exp_width, x_man_width)
    if type_in0 not in writer.types:
        writer.type_buff += new_bfp_ty(x_row, x_col, x_exp_width, x_man_width)
        writer.types.append(type_in0)

    type_in1 = get_bfp_ty(x_row, x_col, w_exp_width, w_man_width)
    if type_in1 not in writer.types:
        writer.type_buff += new_bfp_ty(x_row, x_col, w_exp_width, w_man_width)
        writer.types.append(type_in1)

    type_out = get_bfp_ty(y_row, y_col, ew, mw)
    if type_out not in writer.types:
        writer.type_buff += new_bfp_ty(y_row, y_col, ew, mw)
        writer.types.append(type_out)

    body = f"""
ap_uint<{ew}> max_exp = 0;
"""

    for i in range(0, x_row):
        for j in range(0, x_col):
            body += f"""
ap_uint<1> sign_{i}_{j}_0 = d0.data_{i}_{j}[{x_man_width}];
ap_uint<{ew}> exp_{i}_{j}_0 = d0.exponent;
ap_uint<{mw}> man_{i}_{j}_0 = d0.data_{i}_{j}.range({x_man_width}, 0);
ap_uint<1> sign_{i}_{j}_1 = d0.data_{i}_{j}[{x_man_width}];
ap_uint<{ew}> exp_{i}_{j}_1 = d0.exponent;
ap_uint<{mw}> man_{i}_{j}_1 = d0.data_{i}_{j}.range({x_man_width}, 0);
ap_uint<1> sign_{i}_{j}_2;
ap_uint<{ew}> exp_{i}_{j}_2;
ap_uint<{mw}> man_{i}_{j}_2;
bfp_adder_{adder_id}(sign_{i}_{j}_0, exp_{i}_{j}_0, man_{i}_{j}_0, sign_{i}_{j}_1, exp_{i}_{j}_1, man_{i}_{j}_1, &sign_{i}_{j}_2, &exp_{i}_{j}_2, &man_{i}_{j}_2);

ap_uint<{ew}> res_{i}_{j}_exp = exp_{i}_{j}_2;
ap_int<{mw+1}> res_{i}_{j}_man = (sign_{i}_{j}_2, man_{i}_{j}_2);
max_exp = (max_exp > exp_{i}_{j}_2) ? max_exp : exp_{i}_{j}_2; 
"""
    body += "data->exponent = max_exp;"

    for i in range(0, x_row):
        for j in range(0, x_col):
            body += f"""
data->data_{i}_{j} = res_{i}_{j}_man >> (max_exp - res_{i}_{j}_exp);
"""
    op_id = writer.op_id
    buff = f"""
// BFP Add Block:
void bfp_block_adder_{op_id}({type_in0} d0, 
{type_in1} d1, 
{type_out} *data) {{
#pragma HLS INLINE
{body}
}}
"""
    writer.code_buff += buff
    writer.op_id += 1

    return writer
