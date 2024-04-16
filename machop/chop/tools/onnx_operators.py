import torch

"""
    This module contains a collection of ONNX operators implemented
        using Pytorch primitives.
"""


def onnx_gemm(A, B, C=None, alpha=1.0, beta=1.0, transA=False, transB=False):
    # Transpose matrices A and B if needed
    A = A.transpose() if transA else A
    B = B.transpose() if transB else B

    # Perform matrix multiplication
    result = alpha * torch.matmul(A, B)

    # Add optional matrix C
    if C is not None:
        result += beta * C

    return result


def onnx_slice(data, starts, ends, axes=None, steps=None):
    assert len(starts) == len(ends), "Starts and ends must have the same length"
    starts = starts.to(torch.int64)
    ends = ends.to(torch.int64)

    rank = len(data.shape)

    if axes is None:
        axes = list(range(rank))
    else:
        axes = axes.to(torch.int64)

    if steps is None:
        steps = [1] * rank
    else:
        steps = steps.to(torch.int64)

    # Default slices define entire range in each dimension
    slices = [slice(0, data.shape[i], 1) for i in range(rank)]
    for idx, dim in enumerate(axes):
        slices[dim] = slice(starts[idx], ends[idx], steps[idx])

    return data[slices]


def onnx_unsqueeze(input, dim):
    for i in dim:
        input = torch.unsqueeze(input, i)
    return input


def onnx_gather(input, dim, index):
    """Gather operator with support for broadcasting.
    See https://github.com/pytorch/pytorch/issues/9407

    Args:
        input (_type_): _description_
        dim (_type_): _description_
        index (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(list(input))

    n_dims = len(input.shape)
    idx_list = [
        torch.arange(input.shape[i])[(None,) * i + (...,) + (None,) * (n_dims - i - 1)]
        for i in range(n_dims)
    ]
    idx_list[dim] = index.squeeze()[
        (None,) * dim + (...,) + (None,) * (n_dims - dim - 1)
    ]
    return input[idx_list]


def onnx_shape(input):
    return torch.Tensor([i for i in input.shape])


def onnx_reshape(input, shape):
    if isinstance(shape, torch.Tensor):
        shape = tuple(shape.to(torch.int64).tolist())
    return torch.reshape(input, shape)
