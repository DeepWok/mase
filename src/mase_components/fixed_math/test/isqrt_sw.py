import math


def find_msb(x: int, width: int) -> int:
    msb_index = width - 1
    for i in range(1, width + 1):
        power = 2 ** (width - i)
        if power <= x:
            return width - i
    return msb_index


def float_to_int(x: float, int_width: int, frac_width: int) -> int:
    integer = int(x)
    x -= integer
    res = integer * (2**frac_width)
    for i in range(1, frac_width + 1):
        power = 2 ** (-i)
        if power <= x:
            x -= power
            res += 2 ** (frac_width - i)
    return res


def int_to_float(x: int, int_width: int, frac_width: int) -> float:
    integer = x / (2**frac_width)
    fraction = x - integer * 2**frac_width
    res = integer

    for i in range(1, frac_width + 1):
        power = 2 ** (frac_width - i)
        if power < fraction:
            res += 2 ** (-i)
            fraction -= power
    return res


def range_reduction_sw(x: int, width: int) -> int:
    """model of range reduction for isqrt"""
    # Find MSB
    # NOTE: if the input is 0 then consider msb index as width-1.
    msb_index = find_msb(x, width)
    res = x
    if msb_index < (width - 1):
        res = x * 2 ** (width - 1 - msb_index)
    return res


def range_augmentation_sw(
    x_red: int, msb_index: int, width: int, frac_width: int
) -> int:
    const_len = 16
    ISQRT2 = float_to_int(1 / math.sqrt(2), 1, const_len - 1)
    SQRT2 = float_to_int(math.sqrt(2), 1, const_len - 1)
    """model of range augmentation for isqrt"""
    shifted_amount = frac_width - msb_index
    shift_amount = None
    res = None

    if shifted_amount > 0:
        if shifted_amount % 2 == 0:
            shift_amount = shifted_amount // 2
            res = x_red
        else:
            shift_amount = (shifted_amount - 1) // 2
            res = (x_red * SQRT2) >> (const_len - 1)
        res = res * 2 ** (shift_amount)
    elif shifted_amount < 0:
        if shifted_amount % 2 == 0:
            shift_amount = -shifted_amount // 2
            res = x_red
        else:
            shift_amount = (-shifted_amount - 1) // 2
            res = x_red * ISQRT2 // 2 ** (const_len - 1)
        res = res // 2 ** (shift_amount)
    else:
        res = x_red
    res = res >> (width - 1 - frac_width)
    return res


def fixed_lut_index_sw(x_red: int, width: int, lut_pow: int) -> int:
    """model for finding the lut index for lut isqrt value"""
    if width == 1 or x_red == 0:
        res = 0
    else:
        res = x_red - 2 ** (width - 1)
    res = res * 2**lut_pow
    res = res / 2 ** (width - 1)
    # FORMAT OUTPUT: Q(WIDTH).0
    return int(res)


def make_lut(lut_size, width):
    lut_step = 1 / (lut_size + 1)
    x = 1 + lut_step
    lut = []
    for i in range(lut_size):
        value = 1 / math.sqrt(x)
        value = float_to_int(value, 1, width - 1)
        lut.append(value)
        x += lut_step

    return lut


def nr_stage_sw(x_red: int, in_width: int, initial_guess: int) -> int:
    """model of newton raphson stage"""
    # NOTE: if width is 1 then set output to 0 always because this part gets ignored by logic.
    if in_width < 2:
        threehalfs = 0
    else:
        threehalfs = 3 * 2 ** (in_width - 2)

    y = initial_guess
    x_red = x_red >> 1

    yy = (y * y) >> (in_width - 1)
    mult = (yy * x_red) >> (in_width - 1)
    sub = threehalfs - mult
    y = (y * sub) >> (in_width - 1)

    return y


def isqrt_sw2(
    x: int, in_width: int, frac_width: int, lut_pow: int, lut: list, debug=False
) -> int:
    int_width = in_width - frac_width
    MAX_NUM = (1 << in_width) - 1

    if x == 0:
        return MAX_NUM
    msb_index = find_msb(x, in_width)

    x_red = range_reduction_sw(x, in_width)
    if debug:
        print("MSB index: ", msb_index)
        print("X red: ", int_to_float(x_red, 1, in_width - 1))

    ONE = float_to_int(1, 1, in_width - 1)
    if x_red == ONE:
        out = range_augmentation_sw(x_red, msb_index, in_width, frac_width)
        if debug:
            print("OUT: ", int_to_float(out, int_width, frac_width))
        if out > MAX_NUM:
            if debug:
                print("MAX NUM")
            return MAX_NUM
        return out
    lut_index = fixed_lut_index_sw(x_red, in_width, lut_pow)
    if lut_index > 31:
        print("X: ", x)
        print("MSB index: ", msb_index)
        print("X red: ", int_to_float(x_red, 1, in_width - 1))
        print("INT WIDTH: ", int_width)
        print("FRAC WIDTH: ", frac_width)
    initial_guess = lut[lut_index]

    y = nr_stage_sw(x_red, in_width, initial_guess)
    y = range_augmentation_sw(y, msb_index, in_width, frac_width)

    if debug:
        print("LUT index: ", lut_index)
        print("LUT value: ", int_to_float(initial_guess, 1, in_width - 1))
        print("YY       : ", int_to_float(yy, 1, in_width))
        print("MULT     : ", int_to_float(mult, 1, in_width))
        print("SUB      : ", int_to_float(sub, 1, in_width))
        print("Result   : ", int_to_float(y, int_width, frac_width))

    if y > MAX_NUM:
        return MAX_NUM
    return y


def single_test(
    val: int, verbose: bool, int_width, frac_width, lut_pow, lut, debug=False
) -> float:
    val_f = int_to_float(val, int_width, frac_width)
    width = int_width + frac_width
    MAX_NUM_INT = (1 << width) - 1
    MAX_NUM_FLOAT = int_to_float(MAX_NUM_INT, int_width, frac_width)
    expected_f = None
    if val_f == 0:
        expected_f = MAX_NUM_FLOAT
    else:
        expected_f = 1 / math.sqrt(val_f)
    if expected_f > MAX_NUM_FLOAT:
        expected_f = MAX_NUM_FLOAT
    expected_int = float_to_int(expected_f, int_width, frac_width)
    expected_f = int_to_float(expected_int, int_width, frac_width)

    output = isqrt_sw2(val, width, frac_width, lut_pow, lut, debug)
    output_f = int_to_float(output, int_width, frac_width)
    error = abs(expected_f - output_f)

    if verbose:
        print(f"sqrt({val_f}) = {output_f} | Exp: {expected_f} | Error: {error}")
    return error


def test_sw_model_format(num_bits, sweep, int_width, frac_width, lut_pow):
    max_error = 0
    allowed_error = 2 ** (-frac_width) * num_bits
    width = int_width + frac_width
    lut_size = 1 << lut_pow
    lut = make_lut(lut_size, width)
    verbose = False
    for val in range(sweep):
        error = single_test(val, verbose, int_width, frac_width, lut_pow, lut)
        max_error = max(error, max_error)
        if error > allowed_error:
            print(
                f"""
FAIL

Input:
X         : {val}
INT WIDTH : {int_width}
FRAC WIDTH: {frac_width}

Max error allowed:
{allowed_error}

Max error observed:
{max_error}
                    """
            )
            print("ERROR Excedded!")
            return -1
    print(
        f"""
PASS
Test: Q{int_width}.{frac_width}

Max error allowed:
{allowed_error}

Max error observed:
{max_error}
            """
    )


def test_isqrt_sw_model():
    num_bits = 3  # Number of error bits.
    for frac_width in range(1, 9):
        for int_width in range(1, 9):
            width = int_width + frac_width
            sweep = (1 << width) - 1
            lut_pow = 5
            error_code = test_sw_model_format(
                num_bits, sweep, int_width, frac_width, lut_pow
            )
            if error_code == -1:
                return


def debug_single():
    lut_pow = 5
    lut_size = 2**lut_pow
    int_width = 2
    frac_width = 1
    width = int_width + frac_width
    lut = make_lut(lut_size, width)
    val = 1
    verbose = True
    debug = True
    error = single_test(val, verbose, int_width, frac_width, lut_pow, lut, debug)


def lut_parameter_dict(lut_size: int, width: int, lut_prefix: str = "LUT"):
    lut = make_lut(lut_size, width)
    parameters = {}
    for i in range(lut_size):
        if i < 10:
            lut_suffix = "0" + str(i)
        else:
            lut_suffix = str(i)
        name = lut_prefix + lut_suffix
        parameters |= {name: lut[i]}
    return parameters


if __name__ == "__main__":
    # debug_single()
    test_isqrt_sw_model()
