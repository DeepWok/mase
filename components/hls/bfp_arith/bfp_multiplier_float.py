def bfp_multiplier_float_gen(
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

    buff = f"""
// BFP Single Multiplier_float:
{fpt} bfp_multiplier_float_{op_id}(ap_uint<{x_exp_width}> exp_0, ap_uint<{x_man_width+1}> man_0, ap_uint<{w_exp_width}> exp_1, ap_uint<{w_man_width+1}> man_1) {{
#pragma HLS INLINE 

ap_int<{y_exp_width}> s0 = (man_0[{x_man_width}], exp_0);
ap_int<{y_exp_width}> s1 = (man_1[{w_man_width}], exp_1);
ap_int<{y_exp_width}> s2 = s0 + s1;

ap_uint<{y_man_width}> e2 = man_0.range({x_man_width}, 0) * man_1.range({w_man_width}, 0);
// TODO: Not sure if the bit index is correct
if (e2[1]){{
    s2++;
    e2 >>= 1;
}}

ap_uint<1> sign_2 = man_0[{x_man_width}] ^ man_1[{w_man_width}];
ap_uint<{ew}> exp_2 = s2.range({y_exp_width-1}, 0);
ap_uint<{mw}> man_2 = e2;
ap_uint<{1+ew+mw}> val = (sign_2, exp_2, man_2);

{union_ty} u;
u.intval = val;
return u.fpval;
}}
"""
    writer.code_buff += buff
    writer.op_id += 1

    return writer
