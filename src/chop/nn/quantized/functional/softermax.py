from torch import Tensor
from math import ceil

from chop.nn.quantizers import (
    integer_quantizer,
    integer_floor_quantizer,
)
from chop.nn.functional import softermax


def fixed_softermax(
    input: Tensor, q_config: dict = None, out_q_config: dict = None, dim: int = 0
) -> Tensor:
    """Fixed-point softermax implementation, according to "Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers" paper (https://arxiv.org/abs/2103.09301).

    Args:
        input (Tensor): Input tensor

    Returns:
        Tensor: Output tensor
    """
    if q_config is not None:
        input = integer_quantizer(input, **q_config)

    out = softermax(input, dim=dim)

    if out_q_config is not None:
        out = integer_floor_quantizer(out, **out_q_config)

    return out
