from functools import partial

from ....graph.mase_tracer import mark_as_leaf_func
from ..quantizers import integer_quantizer


@mark_as_leaf_func
def integer_add(x, y, config):
    bypass = config.get("bypass", False)
    # establish quantizers
    x_width, x_frac_width = config["input_width"], config["input_frac_width"]
    x_quantizer = partial(integer_quantizer, width=x_width, frac_width=x_frac_width)

    x = x_quantizer(x)
    y = x_quantizer(y)
    return x + y
