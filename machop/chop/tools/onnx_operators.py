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


def onnx_squeeze(input, dim):
    if isinstance(dim, torch.nn.parameter.Parameter):
        dim = dim.item()
    return torch.squeeze(input, dim)


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

    # expand_shape = list(index.shape[:-1]) + list(input.shape)
    # tmp_inp = input.expand(expand_shape)

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


def onnx_identity(input):
    return input


def onnx_expand(input, size):
    if isinstance(size, torch.Size):
        size = tuple(size)
    elif isinstance(size, torch.Tensor):
        size = tuple(size.to(torch.int64).tolist())
    return input.expand(size=size)


def onnx_where(condition, input, other):
    cond = condition
    pre_input_shape = input.shape
    pre_other_shape = other.shape

    if len(input.shape) == 0:
        input = input.unsqueeze(dim=0)

    # Two-way broadcasting of input tensors
    input, other = torch.broadcast_tensors(input, other)

    assert (
        condition.shape == input.shape == other.shape
    ), "Condition tensor has incorrect shape."

    # Convert condition to a boolean tensor
    condition = torch.where(
        condition == 0,
        torch.full(input.shape, False, dtype=torch.bool),
        torch.full(input.shape, True, dtype=torch.bool),
    ).to(torch.bool)
    return torch.where(condition, input, other)


def onnx_full(size, fill_value):
    if isinstance(size, torch.Tensor):
        size = tuple(size.to(torch.int64).tolist())
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    return torch.full(size, fill_value)


def onnx_min(*args, **kwargs):
    input = torch.broadcast_tensors(*kwargs["input"])
    if len(input) <= 1:
        raise ValueError(f"Expected 2 or more inputs, but received {len(input)}.")

    # minimum only accepts two inputs, so maintain a running minimum
    result = input[0]
    for i in range(1, len(input)):
        result = torch.minimum(result, input[i])
    return result


def onnx_permute(input, dims):
    input = input.squeeze()
    if dims is None:
        dims = [i for i in reversed(range(len(input.shape)))]
    return torch.permute(input, dims)
