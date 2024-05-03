from torch import Tensor
from math import ceil

from chop.passes.graph.transforms.quantize.quantizers import integer_quantizer
from chop.nn.functional import softermax


def fixed_softermax(input: Tensor, q_config: dict) -> Tensor:
    """Fixed-point softermax implementation, according to "Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers" paper (https://arxiv.org/abs/2103.09301).

    Args:
        input (Tensor): Input tensor

    Returns:
        Tensor: Output tensor
    """
    input = integer_quantizer(input, **q_config)
    return softermax(input)
