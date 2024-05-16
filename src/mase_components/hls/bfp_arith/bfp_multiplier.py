def bfp_multiplier_gen(
    writer,
    x_exp_width=16,
    x_man_width=8,
    w_exp_width=16,
    w_man_width=8,
):
    """
    This script generates a element-level bfp mult in HLS
    """
    assert writer is not None
    y_exp_width = max(x_exp_width, w_exp_width) + 1
    y_man_width = x_man_width * w_man_width

    op_id = writer.op_id
    buff = f"""
// BFP Single Multiplier:
void bfp_multiplier_{op_id}(ap_uint<1> sign_0, ap_uint<{x_exp_width}> exp_0, ap_uint<{x_man_width}> man_0, ap_uint<1> sign_1, ap_uint<{w_exp_width}> exp_1, ap_uint<{w_man_width}> man_1, ap_uint<1> *sign_2, ap_uint<{y_exp_width}> *exp_2, ap_uint<{y_man_width}> *man_2) {{
#pragma HLS INLINE 

ap_int<{y_exp_width}> s0 = (sign_0, exp_0);
ap_int<{y_exp_width}> s1 = (sign_1, exp_1);
ap_int<{y_exp_width}> s2 = s0 + s1;

ap_uint<{y_man_width}> e2 = man_0 * man_1;
// TODO: Not sure if the bit index is correct
if (e2[1]){{
    s2++;
    e2 >>= 1;
}}

*sign_2 = sign_0 ^ sign_1;
*exp_2 = s2.range({y_exp_width-1}, 0);
*man_2 = e2;
}}
"""
    writer.code_buff += buff
    writer.op_id += 1

    return writer
