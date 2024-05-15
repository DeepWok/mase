def clog2(x):
    return int(math.ceil(math.log2(x)))


def get_fixed_ty(row, col, w, fw):
    return f"fixed_{row}_{col}_{w}_{fw}_t"


def new_fixed_ty(row, col, w, fw):
    buff = f"struct {get_fixed_ty(row, col, w, fw)} {{"
    for i in range(0, row):
        for j in range(0, col):
            buff += f"ap_fixed<{w}, {w-fw}> data_{i}_{j};"
    buff += "};"
    return buff
