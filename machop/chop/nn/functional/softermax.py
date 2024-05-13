from torch import Tensor


def softermax(input: Tensor, dim: int) -> Tensor:
    """Softermax implementation, according to "Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers" paper (https://arxiv.org/abs/2103.09301).

    Args:
        input (Tensor): Input tensor

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
