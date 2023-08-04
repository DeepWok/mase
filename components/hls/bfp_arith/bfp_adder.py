def bfp_adder_gen(
    writer,
    exp_width=16,
    man_width=8,
):
    """
    This script generates a element-level bfp add in HLS
    """
    assert writer is not None
    ew = exp_width
    mw = man_width
    assert (ew == 8 and mw == 23) or (ew == 5 and mw == 10)

    # Add union
    if ew == 8:
        # Use fp32
        fpt = "float"
        union_ty = """union
{
  unsigned int intval;
  float fpval;
}
"""
    else:
        # Use fp16
        fpt = "half"
        union_ty = """union
{
  unsigned int intval;
  half fpval;
}
"""

    op_id = writer.op_id
    buff = f"""
// BFP Single Adder:
void bfp_adder_{op_id}(ap_uint<1> sign_0, ap_uint<{ew}> exp_0, ap_uint<{mw}> man_0, ap_uint<1> sign_1, ap_uint<{ew}> exp_1, ap_uint<{mw}> man_1, ap_uint<1> *sign_2, ap_uint<{ew}> *exp_2, ap_uint<{mw}> *man_2) {{
#pragma HLS INLINE 
{union_ty} union_0;
{union_ty} union_1;

ap_uint<{ew+mw+1}> bits_0 = (sign_0, exp_0, man_0);
union_0.intval = bits_0;
{fpt} fp_data_0 = union_0.fpval;

ap_uint<{ew+mw+1}> bits_1 = (sign_1, exp_1, man_1);
union_1.intval = bits_1;
{fpt} fp_data_1 = union_1.fpval;

{fpt} fp_data_2 = fp_data_0 + fp_data_1;

union
{{
  unsigned int intval;
  half fpval;
}} union_2;
union_2.fpval = fp_data_2;
ap_uint<{ew+mw+1}> bits_2 = union_2.intval;
(*sign_2, *exp_2, *man_2) = bits_2;
}}
"""
    writer.code_buff += buff
    writer.op_id += 1

    return writer
