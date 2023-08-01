from math import ceil, log2
from typing import List, Tuple, Union
from numpy import ndarray
from torch import Tensor


from .utils import (
    ternarised_scaled_op,
    ternarised_op,
)


def ternary_quantizer(
    x: Union[Tensor, ndarray],
    threshold: float = None,
    maximum: float = None,
    median: float = None,
    mean: float = None,
    scaling_factor: bool = False,
):
    """
    Required: at least one of either threshold, maximum, median, or mean
    - Do ternary quantization to input
    - Optionally do equal_distance quantization

    ---
    - forward:
    - backward: STE

    ---
    scaling_factor: use ternarization with scaling

    ---
    Refer to https://arxiv.org/pdf/1807.07948.pdf

    """
    if threshold is None:
        if mean:
            threshold = 0.75 * mean  # https://arxiv.org/pdf/1605.04711.pdf
        elif median:
            threshold = 0.75 * median  # fallback if mean not available
        elif maximum:
            threshold = 0.05 * maximum  # https://arxiv.org/pdf/1807.07948.pdf
        else:
            raise RuntimeError(
                "No appropriate ternary quantisation threshold is determinable! Did you run statistical analysis pass?"
            )
    if scaling_factor:
        x = ternarised_scaled_op(
            x, threshold
        )  # [mean, 0 ,-mean] # this function determines the mean on the fly, maybe we could make an alternative which uses the metadata?
    else:
        x = ternarised_op(x, threshold)  # [1, 0 ,-1]
    return x
