from torch import Tensor
from math import ceil


def softermax(input: Tensor) -> Tensor:
    """Softermax implementation, according to "Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers" paper (https://arxiv.org/abs/2103.09301).

    Args:
        input (Tensor): Input tensor

    Returns:
        Tensor: Output tensor
    """
    powers = 2 ** (input - ceil(input.max()))
    return powers / powers.sum()
