import torch
from torch import Tensor


def softermax(x: Tensor, dim: int) -> Tensor:
    """Softermax implementation, according to "Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers" paper (https://arxiv.org/abs/2103.09301).

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Output tensor
    """
    input = input.squeeze()
    out = input - input.max(dim=1).values.floor()
    out = 2**out
    row_sum = out.sum(dim=1).reshape((-1, 1)).expand(input.shape)
    # Elementwise division
    out = out / row_sum
    return out


# Derek: Temp!
def softermax2(x: Tensor, dim: int) -> Tensor:
    out = x - torch.floor(x.max(dim=dim, keepdim=True).values)
    out = 2**out
    row_sum = out.sum(dim=dim, keepdim=True)
    out = out / row_sum
    return out
