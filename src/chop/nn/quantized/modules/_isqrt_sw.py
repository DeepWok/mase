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


def range_reduction_sw(x: int, width: int) -> int:
    msb_index = find_msb(x, width)
    if msb_index < (width - 1):
        return x * 2 ** (width - 1 - msb_index)
    return x


def range_augmentation_sw(
    x_red: int, msb_index: int, width: int, frac_width: int
) -> int:
    const_len = 16
    isqrt2 = float_to_int(1 / math.sqrt(2), 1, const_len - 1)
    sqrt2 = float_to_int(math.sqrt(2), 1, const_len - 1)
    shifted_amount = frac_width - msb_index

    if shifted_amount > 0:
        if shifted_amount % 2 == 0:
            shift_amount = shifted_amount // 2
            res = x_red
        else:
            shift_amount = (shifted_amount - 1) // 2
            res = (x_red * sqrt2) >> (const_len - 1)
        res = res * 2**shift_amount
    elif shifted_amount < 0:
        if shifted_amount % 2 == 0:
            shift_amount = -shifted_amount // 2
            res = x_red
        else:
            shift_amount = (-shifted_amount - 1) // 2
            res = x_red * isqrt2 // 2 ** (const_len - 1)
        res = res // 2**shift_amount
    else:
        res = x_red

    return res >> (width - 1 - frac_width)


def fixed_lut_index_sw(x_red: int, width: int, lut_pow: int) -> int:
    if width == 1 or x_red == 0:
        res = 0
    else:
        res = x_red - 2 ** (width - 1)
    res = res * 2**lut_pow
    res = res / 2 ** (width - 1)
    return int(res)


def nr_stage_sw(x_red: int, in_width: int, initial_guess: int) -> int:
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
    x: int, in_width: int, frac_width: int, lut_pow: int, lut: list, debug: bool = False
) -> int:
    del debug
    max_num = (1 << in_width) - 1

    if x == 0:
        return max_num

    msb_index = find_msb(x, in_width)
    x_red = range_reduction_sw(x, in_width)

    one = float_to_int(1, 1, in_width - 1)
    if x_red == one:
        out = range_augmentation_sw(x_red, msb_index, in_width, frac_width)
        return min(out, max_num)

    lut_index = fixed_lut_index_sw(x_red, in_width, lut_pow)
    initial_guess = lut[lut_index]

    y = nr_stage_sw(x_red, in_width, initial_guess)
    y = range_augmentation_sw(y, msb_index, in_width, frac_width)
    return min(y, max_num)
