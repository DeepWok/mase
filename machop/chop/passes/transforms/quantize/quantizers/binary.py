from math import ceil, log2
from typing import List, Tuple, Union
import numpy
import torch


from .utils import (
    my_clamp,
    my_round,
    my_clamp,
    binarised_bipolar_op,
    binarised_zeroOne_op,
    binarised_zeroScaled_op,
    ternarised_scaled_op,
    ternarised_op,
)


def binary_quantizer(
    x: torch.Tensor | numpy.ndarray, stochastic: bool = False, bipolar: bool = False
):
    """
    - Do binary quantization to input
    - Optionally do stochastic quantization

    ---
    - forward:
    - backward: STE

    ---
    stochastic: enable stochastic quantization otherwise threshold is defaulted to 0
    positive_binarized: binarized input to {0, -1} if enabled else binarized input to {-1, 1}

    ---
    Refer to https://arxiv.org/pdf/1603.01025.pdf

    """

    if stochastic:
        x_sig = my_clamp((x + 1) / 2, 0, 1)
        x_rand = (
            torch.rand_like(x)
            if isinstance(x, torch.Tensor)
            else numpy.random.rand(*x.shape)
        )

        x = (
            binarised_bipolar_op(x_sig, x_rand)
            if bipolar
            else binarised_zeroScaled_op(x_sig, x_rand)
        )
    else:
        x = binarised_bipolar_op(x, 0) if bipolar else binarised_zeroScaled_op(x, 0)

    return x
