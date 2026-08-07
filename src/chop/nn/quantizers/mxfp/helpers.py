"""Shape helpers shared by MX-format quantizers."""

import torch
from torch import Tensor


def block_rows_for_quantize(
    tensor: Tensor,
    block_dim: int,
    block_size: int,
) -> tuple[Tensor, int]:
    """Return independently padded rows with the block axis last.

    Zero padding is local to each logical row and therefore cannot merge the
    tail of one attention head with the start of another.
    """
    if tensor.ndim == 0:
        raise ValueError("MX block quantization requires at least one dimension")
    if block_size <= 0:
        raise ValueError("MX block size must be positive")
    ori_shape = tuple(tensor.shape)
    ndim = len(ori_shape)
    block_dim = block_dim % ndim

    permute = list(range(ndim))
    permute.append(permute.pop(block_dim))
    rows = tensor.permute(permute).contiguous()
    axis_size = rows.shape[-1]
    padded_axis_size = ((axis_size + block_size - 1) // block_size) * block_size
    if padded_axis_size != axis_size:
        padded_shape = (*rows.shape[:-1], padded_axis_size)
        padded = torch.zeros(padded_shape, dtype=rows.dtype, device=rows.device)
        padded[..., :axis_size] = rows
        rows = padded
    return rows.reshape(-1, block_size), padded_axis_size


def restore_quantized_rows(
    block_tensor: Tensor,
    ori_shape: tuple[int, ...],
    block_dim: int,
    padded_axis_size: int,
) -> Tensor:
    """Remove row-local padding and restore the original dimension order."""
    ndim = len(ori_shape)
    block_dim = block_dim % ndim

    permuted_shape = list(ori_shape)
    permuted_shape.append(permuted_shape.pop(block_dim))
    axis_size = permuted_shape[-1]
    padded_shape = (*permuted_shape[:-1], padded_axis_size)
    tensor = block_tensor.reshape(padded_shape)[..., :axis_size]

    inverse_permute = list(range(ndim))
    inverse_permute.insert(block_dim, inverse_permute.pop(-1))
    return tensor.permute(inverse_permute).contiguous()
