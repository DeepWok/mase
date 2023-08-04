def clog2(x):
    return int(math.ceil(math.log2(x)))


def get_bfp_ty(row, col, ew, mw):
    return f"fixed_{row}_{col}_{ew}_{mw}_t"


def new_bfp_ty(row, col, ew, mw):
    buff = f"struct {get_bfp_ty(row, col, ew, mw)} {{"
    buff += f"ap_uint<{ew}> exponent;"
    for i in range(0, row):
        for j in range(0, col):
            buff += f"ap_int<{mw+1}> data_{i}_{j};"
    buff += "};"
    return buff
