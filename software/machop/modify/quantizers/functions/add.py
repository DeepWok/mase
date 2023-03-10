from functools import partial
from ..quantizers import integer_quantizer


def integer_add(x, y, config):
    bypass = config.get("bypass", False)
    # establish quantizers
    x_bits, x_bias = config["input_bits"], config["input_bias"]
    x_quantizer = partial(integer_quantizer, bits=x_bits, bias=x_bias)

    x = x_quantizer(x)
    y = x_quantizer(y)
    return x + y
