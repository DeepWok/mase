from functools import partial

import torch

from ....graph.mase_tracer import mark_as_leaf_func
from ..quantizers import integer_quantizer


@mark_as_leaf_func
def integer_matmul(x, y, config):
    output_width, output_frac_width = (
        config["output_width"],
        config["output_frac_width"],
    )
    output_quantizer = partial(
        integer_quantizer, width=output_width, frac_width=output_frac_width
    )

    return output_quantizer(torch.matmul(x, y))
