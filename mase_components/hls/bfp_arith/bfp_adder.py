from .utils import clog2, get_bfp_ty, new_bfp_ty


def bfp_adder_gen(
    writer,
    x_exp_width=16,
    x_man_width=8,
    w_exp_width=16,
    w_man_width=8,
):
    """
    This script generates a element-level bfp add in HLS
    """
    assert writer is not None

    if x_exp_width > w_exp_width:
        y_exp_width = x_exp_width
        y_man_width = x_man_width
    else:
        y_exp_width = w_exp_width
        y_man_width = w_man_width

    min_shift = min(x_man_width, w_man_width)

    true_body = ""
    false_body = ""
    for i in range(1, min_shift):
        true_body += f"""
if (diff == {i}) {{
(sign_2_temp, carry, man_2_temp) = (sign_0, zero, man_0) + (sign_1, zero, (man_1 >> {i}));
exp_2_temp = exp_0 + carry;
}}
"""
        false_body += f"""
if (diff == {i}) {{
(sign_2_temp, carry, man_2_temp) = (sign_0, zero, (man_0 >> {i})) + (sign_1, zero, man_1);
exp_2_temp = exp_1 + carry;
}}
"""

    op_id = writer.op_id
    buff = f"""
// BFP Single Adder:
void bfp_adder_{op_id}(ap_uint<1> sign_0, ap_uint<{x_exp_width}> exp_0, ap_uint<{x_man_width}> man_0, ap_uint<1> sign_1, ap_uint<{w_exp_width}> exp_1, ap_uint<{w_man_width}> man_1, ap_uint<1> *sign_2, ap_uint<{y_exp_width}> *exp_2, ap_uint<{y_man_width}> *man_2) {{
#pragma HLS INLINE 

ap_uint<{y_exp_width}> exp_2_temp;
ap_uint<1> sign_2_temp;
ap_uint<1> carry;
ap_uint<1> zero = 0;
ap_uint<{y_man_width}> man_2_temp;
if (exp_0 == exp_1) {{
exp_2_temp = exp_0;
(sign_2_temp, man_2_temp) = (sign_0, man_0) + (sign_1, man_1);
}}
else if (exp_0 - exp_1 >= {min_shift}) {{
exp_2_temp = exp_0;
(sign_2_temp, man_2_temp) = (sign_0, man_0);
}}
else if (exp_1 - exp_0 >= {min_shift}) {{
exp_2_temp = exp_1;
(sign_2_temp, man_2_temp) = (sign_1, man_1);
}}
else if (exp_0 > exp_1) {{
ap_uint<{y_exp_width}> diff = exp_0 - exp_1;
{true_body}
}} else {{
ap_uint<{y_exp_width}> diff = exp_1 - exp_0;
{false_body}
}}
*exp_2 = exp_2_temp;
*sign_2 = sign_2_temp;
*man_2 = man_2_temp;
}}
"""
    writer.code_buff += buff
    writer.op_id += 1

    return writer
