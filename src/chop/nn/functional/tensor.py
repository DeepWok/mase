import torch
import torch.fx as fx

# This file contains functional equivalent of some torch.Tensor methods
# which can be casted to call_function nodes by the replace_method_with_function pass.
# They must have the same signature as their torch.Tensor equivalents with an added
# input node at position 0.


@fx.wrap
def torch_size(
    input: torch.Tensor,
    dim: int = None,
):
    return input.size(dim)


@fx.wrap
def torch_expand(
    input: torch.Tensor,
    *sizes,
):
    return input.expand(*sizes)


@fx.wrap
def torch_view(
    input: torch.Tensor,
    *shape,
):
    return input.view(*shape)


@fx.wrap
def torch_contiguous(
    input: torch.Tensor,
    memory_format: torch.memory_format = torch.contiguous_format,
):
    return input.contiguous(memory_format=memory_format)


# The following functions exist in torch functional land,
# however their functional implementation does not accept
# arbitrary argument counts i.e. *args, **kwargs, so we
# reimplement them here.
# ============================================================


@fx.wrap
def torch_reshape(
    input: torch.Tensor,
    *shape,
):
    return input.reshape(*shape)


@fx.wrap
def torch_split(
    input: torch.Tensor,
    split_size: int,
    dim: int = 0,
):
    return input.split(split_size, dim)


@fx.wrap
def torch_permute(
    input: torch.Tensor,
    *dims,
):
    return input.permute(*dims)


@fx.wrap
def torch_transpose(
    input: torch.Tensor,
    dim0: int,
    dim1: int,
):
    return input.transpose(dim0, dim1)
