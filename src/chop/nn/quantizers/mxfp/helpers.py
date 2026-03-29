"""
Helper functions for MX-format quantizers.
"""

from torch import Tensor


def flatten_for_quantize(tensor: Tensor, block_dim: int) -> Tensor:
    """
    Permute tensor to move block dimension to last position and flatten.

    Args:
        tensor: Input tensor
        block_dim: Dimension to use for blocking

    Returns:
        Flattened tensor with block_dim moved to last position
    """
    ori_shape = tuple(tensor.shape)
    ndim = len(ori_shape)
    block_dim = block_dim % ndim

    # Create permutation to move block_dim to last position
    permute = list(range(ndim))
    permute.append(permute.pop(block_dim))

    tensor = tensor.permute(permute)
    tensor = tensor.flatten()
    return tensor


def permute_for_dequantize(
    flatten_tensor: Tensor,
    ori_shape: tuple[int, ...],
    block_dim: int,
) -> Tensor:
    """
    Reshape flattened tensor back to original shape after dequantization.

    Args:
        flatten_tensor: Flattened tensor from quantization
        ori_shape: Original tensor shape before flattening
        block_dim: Original block dimension

    Returns:
        Tensor restored to original shape
    """
    ndim = len(ori_shape)
    block_dim = block_dim % ndim

    # Create the shape after moving block_dim to last position
    permuted_shape = list(ori_shape)
    permuted_shape.append(permuted_shape.pop(block_dim))

    # Reshape from flattened form to intermediate permuted form
    tensor = flatten_tensor.reshape(permuted_shape)

    # Create inverse permutation to restore original dimension order
    inverse_permute = list(range(ndim))
    inverse_permute.insert(block_dim, inverse_permute.pop(-1))

    tensor = tensor.permute(inverse_permute)
    return tensor
