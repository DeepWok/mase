from logging import getLogger
from math import ceil
from typing import List

import torch
from torch import Tensor
from torch.autograd.function import InplaceFunction
from torch.nn import functional as F

logger = getLogger(__name__)


# Forced torch gradient overrider
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyFloor(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryBipolar(InplaceFunction):
    """A PyTorch function for binarizing input values.

    This function takes an input tensor and a threshold value and binarizes the input values,
    setting values greater than or equal to the threshold to 1 and values below the threshold to -1.

    Args:
        ctx (torch.autograd.function._ContextMethodMixin): The context object to store intermediate results.
        input (torch.Tensor): The input tensor to be binarized.
        threshold (float or torch.Tensor): The threshold value for binarization.

    Returns:
        torch.Tensor: The binarized output tensor, where values are either -1 or 1.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        return torch.where(input >= threshold, torch.tensor(1.0), torch.tensor(-1.0))

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class BinaryBipolarScaled(InplaceFunction):
    """A PyTorch function for binarizing input values.

    This function takes an input tensor and a threshold value and binarizes the input values,
    setting values greater than or equal to the threshold to mean and values below the threshold to -mean

    Args:
        ctx (torch.autograd.function._ContextMethodMixin): The context object to store intermediate results.
        input (torch.Tensor): The input tensor to be binarized.
        threshold (float or torch.Tensor): The threshold value for binarization.

    Returns:
        torch.Tensor: The binarized output tensor, where values are either -1 or 1.
    """

    @staticmethod
    def alpha(tensor):  # determine batch means
        absvalue = tensor.abs()
        alpha = absvalue.mean(dim=(1, 2, 3), keepdims=True)
        return alpha.view(-1, 1)

    @staticmethod
    def forward(ctx, input, _threshold):
        alpha = BinaryBipolarScaled.alpha(input)  # contains all averages per batch item

        output = torch.zeros_like(input)  # tracer compatability vs torch.zeros()
        pos_one = torch.where(input > 0, 1.0, 0.0)
        neg_one = pos_one - 1
        out = torch.add(pos_one, neg_one)
        output = out * alpha.view(-1, 1, 1, 1).expand(
            -1, input.size()[1], input.size()[2], input.size()[3]
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class BinaryZeroOne(InplaceFunction):
    """A PyTorch function for binarizing input values.

    This function takes an input tensor and a threshold value and binarizes the input values,
    setting values greater than or equal to the threshold to 1 and values below the threshold to 0.

    Args:
        ctx (torch.autograd.function._ContextMethodMixin): The context object to store intermediate results.
        input (torch.Tensor): The input tensor to be binarized.
        threshold (float or torch.Tensor): The threshold value for binarization.

    Returns:
        torch.Tensor: The binarized output tensor, where values are either 0 or 1.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        return torch.where(input >= threshold, torch.tensor(1.0), torch.tensor(0.0))

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class BinaryZeroScaled(InplaceFunction):
    """A PyTorch function for binarizing input values.

    This function takes an input tensor and a threshold value and binarizes the input values,
    setting values greater than or equal to the threshold to 1 and values below the threshold to 0 but scaled with tensor means.

    Args:
        ctx (torch.autograd.function._ContextMethodMixin): The context object to store intermediate results.
        input (torch.Tensor): The input tensor to be binarized.
        threshold (float or torch.Tensor): The threshold value for binarization.

    Returns:
        torch.Tensor: The binarized output tensor
    """

    @staticmethod
    def alpha(tensor):  # determine batch means
        absvalue = tensor.abs()
        alpha = absvalue.mean(dim=(1, 2, 3), keepdims=True)
        return alpha.view(-1, 1)

    @staticmethod
    def forward(ctx, input, _threshold):
        alpha = BinaryZeroScaled.alpha(input)

        pos_one = torch.where(input > 0, 1.0, 0.0)
        output = pos_one * alpha.view(-1, 1, 1, 1).expand(
            -1, input.size()[1], input.size()[2], input.size()[3]
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class TernaryScaled(InplaceFunction):
    """A PyTorch function for ternary scaling of input values.

    This function takes an input tensor and a threshold value and performs ternary scaling,
    where values greater than the threshold are scaled to the mean of the input tensor, values less than or equal to
    the negative threshold are scaled to negative mean of the input tensor, and values within the range (-threshold, threshold]
    are scaled to 0. The scaling factor is determined by the mean of the input tensor.

    Args:
        ctx (torch.autograd.function._ContextMethodMixin): The context object to store intermediate results.
        input (torch.Tensor): The input tensor to be ternary scaled.
        threshold (float or torch.Tensor): The threshold value for ternary scaling.

    Returns:
        torch.Tensor: The ternary scaled output tensor, where values are either -mean, 0, or mean.
    """

    @staticmethod
    def delta(tensor):
        n = tensor[0].nelement()  # total feature map elements

        flat = tensor.flatten(1)

        delta = 0.75 * torch.sum(flat.abs(), dim=(1,)) / n

        delta = (
            delta.unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, tensor.size()[1], tensor.size()[2], tensor.size()[3])
        )  # expand to match input dims

        return delta

    @staticmethod
    def alpha(tensor, delta):
        absvalue = tensor.abs()
        truth_value = (absvalue > delta).to(torch.float32)
        truth_num = truth_value.sum(dim=(1, 2, 3))
        alpha = (truth_value.view(1, -1) * absvalue.view(1, -1)).view(
            tensor.size()
        ).sum(dim=(1, 2, 3)) / truth_num
        return alpha.view(-1, 1)

    @staticmethod
    def forward(ctx, input, _threshold):
        delta = TernaryScaled.delta(input)
        alpha = TernaryScaled.alpha(input, delta)

        output = torch.zeros_like(input)
        pos_one = torch.where(
            input > delta,
            1.0,
            0.0,
        )
        neg_one = torch.where(
            input < -delta,
            -1.0,
            0.0,
        )
        output = (pos_one + neg_one) * alpha.view(-1, 1, 1, 1).expand(
            -1, input.size()[1], input.size()[2], input.size()[3]
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class Ternary(InplaceFunction):
    """A PyTorch function for ternary scaling of input values.

    This function takes an input tensor and a threshold value and performs ternary scaling,
    where values greater than the threshold are 1, values less than or equal to
    the -1, and values within the range (-threshold, threshold]
    are scaled to 0. The scaling factor is determined by the mean of the input tensor.

    Args:
        ctx (torch.autograd.function._ContextMethodMixin): The context object to store intermediate results.
        input (torch.Tensor): The input tensor to be ternary scaled.
        threshold (float or torch.Tensor): The threshold value for ternary scaling.

    Returns:
        torch.Tensor: The ternary scaled output tensor, where values are either -1, 0, or 1.
    """

    @staticmethod
    def delta(tensor):
        n = tensor[0].nelement()  # total feature map elements

        flat = tensor.flatten(1)

        delta = 0.75 * torch.sum(flat.abs(), dim=(1,)) / n

        delta = (
            delta.unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, tensor.size()[1], tensor.size()[2], tensor.size()[3])
        )  # expand to match input dims

        return delta

    @staticmethod
    def alpha(tensor, delta):
        absvalue = tensor.abs()
        truth_value = (absvalue > delta).to(torch.float32)
        truth_num = truth_value.sum(dim=(1, 2, 3))
        alpha = (truth_value.view(1, -1) * absvalue.view(1, -1)).view(
            tensor.size()
        ).sum(dim=(1, 2, 3)) / truth_num
        return alpha.view(-1, 1)

    @staticmethod
    def forward(ctx, input, _threshold):
        delta = TernaryScaled.delta(input)
        alpha = TernaryScaled.alpha(input, delta)

        output = torch.zeros_like(input)
        pos_one = torch.where(
            input > delta,
            1.0,
            0.0,
        )
        neg_one = torch.where(
            input < -delta,
            -1.0,
            0.0,
        )
        output = pos_one + neg_one
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class MySign(InplaceFunction):
    """A PyTorch function for obtaining a ternary tensor value according to sign.

    Returns:
        torch.Tensor: The ternary scaled output tensor, where values are either -1, 0, or 1.
    """

    @staticmethod
    def forward(ctx, input):
        return input.sign()  # result [-1,0,1]

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


mu_sign = MySign.apply
binarised_bipolarScaled_op = BinaryBipolarScaled.apply
binarised_bipolar_op = BinaryBipolar.apply
binarised_zeroOne_op = BinaryZeroOne.apply
binarised_zeroScaled_op = BinaryZeroScaled.apply
ternarised_scaled_op = TernaryScaled.apply
ternarised_op = Ternary.apply
my_clamp = MyClamp.apply
my_round = MyRound.apply
my_floor = MyFloor.apply

# --------------------------------
# Block and unblock
# --------------------------------


def _infer_block_shape(x_shape: List[int], block_shape: List[int]):
    """
    Infer a reasonable block shape.
    - right align block_shape with x_shape,
        1. If len(block_shape) > len(x_shape), truncate redundant block_shape dims.
        2. If block_shape.ndim < x_shape.ndim, prepend -1 to block_shape until block_shape.ndim == x_shape.ndim
        - if block_shape[i] < x_shape[i], inferred_block_shape[i] = block_shape[i]
        - if block_shape[i] >= x_shape[i], inferred_block_shape[i] = x_shape[i]
        - if block_shape[i] == -1, inferred_block_shape[i] = x_shape[i]
    """
    x_ndim = len(x_shape)
    block_ndim = len(block_shape)

    if block_ndim >= x_ndim:
        inferred_block_shape = block_shape[-x_ndim:]
    else:
        inferred_block_shape = [-1] * (x_ndim - block_ndim) + block_shape
    for dim_i in range(x_ndim):
        if (
            inferred_block_shape[dim_i] == -1
            or inferred_block_shape[dim_i] > x_shape[dim_i]
        ):
            inferred_block_shape[dim_i] = x_shape[dim_i]
        else:
            inferred_block_shape[dim_i] = inferred_block_shape[dim_i]
    return inferred_block_shape


def _infer_padding_shape(x_shape: List[int], block_shape: List[int]):
    """
    Calculate paddings to make x_shape[i] divisable by block_shape[i]
    """
    pad_diff = []
    for x_shape_dim_i, block_shape_dim_i in zip(x_shape, block_shape):
        if block_shape_dim_i == -1 or x_shape_dim_i < block_shape_dim_i:
            pad_diff += [0, 0]
        else:
            num_blocks_dim_i = ceil(x_shape_dim_i / block_shape_dim_i)
            new_x_dim_i = num_blocks_dim_i * block_shape_dim_i
            pad_diff += [new_x_dim_i - x_shape_dim_i, 0]
    pad_diff = pad_diff[::-1]
    return pad_diff


def _block_1d_bias(x: Tensor, block_shape: List[int]):
    """
    bias shape: [output_features] -> [num_blocks, block_size]

    The bias of nn.Linear, nn.Conv1d, and nn.Conv2d are all 1D tensors

    ---
    x: a bias with bias.ndim == 1
    """
    assert x.ndim == 1
    x_shape = [i for i in x.shape]
    block_shape = _infer_block_shape(x_shape, block_shape)
    pad_diff = _infer_padding_shape(x_shape, block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)
    blocked_x = padded_x.reshape(padded_x_shape[0] // block_shape[0], block_shape[0])
    per_block_max = torch.abs(blocked_x).max(dim=1, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_1d_bias(
    blocked_x: Tensor,
    x_shape_before_blocking: List[int],
):
    """
    blocked bias shape: [num_blocks, block_size] -> [output_features]

    ---
    blocked x: blocked bias with blocked_bias.ndim == 2
    """
    x = blocked_x.flatten()

    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    # print(f"indexes: {indexes}")
    x = x[indexes]
    return x


def _block_2d_activation(x: Tensor, block_shape: List[int]):
    """
    [batch_size, hidden_size] -> [batch_size, num_blocks, block_size[-1]]
    """
    assert x.ndim == 2
    x_shape = [i for i in x.shape]
    one_batch_shape = [1, x_shape[1]]
    block_shape = _infer_block_shape(one_batch_shape, block_shape=block_shape)
    pad_diff = _infer_padding_shape(x_shape, block_shape=block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)
    # [batch_size, hidden_size] -> [batch_size, num_blocks, block_size[-1]]
    blocked_x = padded_x.reshape(
        x_shape[0], padded_x_shape[1] // block_shape[-1], block_shape[-1]
    )
    per_block_max = torch.abs(blocked_x).max(dim=2, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_2d_activation(blocked_x: Tensor, x_shape_before_blocking: List[int]):
    """
    [batch_size, num_blocks, block_size] -> [batch_size, hidden_size]
    """
    x = blocked_x.flatten(1)

    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    # print(f"indexes: {indexes}")
    x = x[indexes]
    return x


def _block_2d_weight(x: Tensor, block_shape: List[int]):
    """
    [in_features, out_features] -> [block_size_0 * block_size_1, num_blocks]

    """
    assert x.ndim == 2
    x_shape = [i for i in x.shape]
    block_shape = _infer_block_shape(x_shape, block_shape)
    pad_diff = _infer_padding_shape(x_shape, block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)

    padded_x = padded_x.unsqueeze(0).unsqueeze(0)
    # [1, 1, in_features, out_features] -> [1, block_size_0 * block_size_1, num_blocks]
    blocked_x = F.unfold(
        padded_x, kernel_size=block_shape, dilation=1, padding=0, stride=block_shape
    )

    # [1, block_size_0 * block_size_1, num_blocks] -> [block_size_0 * block_size_1, num_blocks]
    blocked_x = blocked_x.squeeze(0)
    per_block_max = torch.abs(blocked_x).max(dim=0, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_2d_weight(
    blocked_x: Tensor, x_shape_before_blocking, padded_x_shape, block_shape
):
    """
    [block_size_0 * block_size_1, num_blocks] -> [in_features, out_features]
    """
    # [block_size_0 * block_size_1, num_blocks] -> [1, padded_x_shape[0], padded_x_shape[1]]
    x = F.fold(
        blocked_x,
        output_size=padded_x_shape,  # [padded_in_features, padded_out_features]
        kernel_size=block_shape,  # [block_shape_0, block_shape_1]
        dilation=1,
        padding=0,
        stride=block_shape,
    )

    x = x.squeeze(0)
    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    x = x[indexes]
    # print(f"indexes: {indexes}")
    return x


def _block_3d_activation(x: Tensor, block_shape: List[int]):
    """
    [batch_size, hidden_dim_0, hidden_dim_1] -> [batch_size, block_size_0 * block_size_1, num_blocks]

    ---
    Return blocked_x, per_block_max, padded_x_shape, block_shape
    """
    assert x.ndim == 3
    x_shape = [i for i in x.shape]
    one_batch_shape = [1, *x_shape[1:]]
    block_shape = _infer_block_shape(one_batch_shape, block_shape)  # [1, ...]
    pad_diff = _infer_padding_shape(one_batch_shape, block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)
    padded_x = padded_x.unsqueeze(1)
    # [batch_size, 1, num_tokens, hidden_size] -> [batch_size, block_size_0 * block_size_1, num_blocks]
    blocked_x = F.unfold(
        padded_x,
        kernel_size=block_shape[1:],
        dilation=1,
        padding=0,
        stride=block_shape[1:],
    )

    per_block_max = torch.abs(blocked_x).max(dim=1, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_3d_activation(
    blocked_x: Tensor, x_shape_before_blocking, padded_x_shape, block_shape
):
    # [batch_size, block_size_0 * block_size_1, num_blocks] -> [batch_size, 1, padded_x_shape_1, padded_x_shape_2]
    x = F.fold(
        blocked_x,
        output_size=padded_x_shape[1:],
        kernel_size=block_shape[1:],
        dilation=1,
        padding=0,
        stride=block_shape[1:],
    )
    x = x.squeeze(1)
    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    x = x[indexes]
    # print(f"indexes: {indexes}")
    return x


def block(x: Tensor, block_shape: List[int], skip_first_dim: bool = False):
    """
    - skip_first_dim (bool): If True, block_shape[0] will always take 1.

    ---
    Return (blocked_x, per_block_max, padded_x_shape, block_shape)
    """
    if x.ndim == 1:
        assert (
            skip_first_dim is False
        ), "skip_first_dim must be False for bias to be blocked"
        return _block_1d_bias(x, block_shape)
    elif x.ndim == 2:
        if skip_first_dim:
            return _block_2d_activation(x, block_shape)
        else:
            return _block_2d_weight(x, block_shape)
    elif x.ndim == 3:
        if skip_first_dim:
            return _block_3d_activation(x, block_shape)
        else:
            raise NotImplementedError("block 3d weight is not supported.")
    else:
        raise RuntimeError(f"Unsupported x.ndim = {x.ndim}")


def unblock(
    blocked_x: Tensor,
    x_shape_before_blocking: List[int],
    padded_x_shape,
    block_shape: List[int],
    skipped_first_dim_when_blocking: bool = True,
):
    if len(x_shape_before_blocking) == 1:
        assert (
            skipped_first_dim_when_blocking is False
        ), "first dim of bias can not have been skipped in blocking"
        return _unblock_to_1d_bias(blocked_x, x_shape_before_blocking)
    elif len(x_shape_before_blocking) == 2:
        if skipped_first_dim_when_blocking:
            return _unblock_to_2d_activation(blocked_x, x_shape_before_blocking)
        else:
            return _unblock_to_2d_weight(
                blocked_x,
                x_shape_before_blocking,
                padded_x_shape,
                block_shape,
            )
    elif len(x_shape_before_blocking) == 3:
        if skipped_first_dim_when_blocking:
            return _unblock_to_3d_activation(
                blocked_x, x_shape_before_blocking, padded_x_shape, block_shape
            )
        else:
            raise NotImplementedError("unblock to 3d weight is not supported")
    else:
        raise RuntimeError(
            "Unsupported n.dims ({}) to unblock back".format(
                len(x_shape_before_blocking)
            )
        )


def _block_multi_dim_weight(x: Tensor, block_shape: List[int]):
    """
    [weight_shape_1, weight_shape_2, ..., weight_shape_n] -> [block_size_1 * block_size_2 * ... * block_size_n, num_blocks]
    """
    # !: Currently this general function is not available because current pytorch (2.0.0) nn.functional.fold does not support input dimension higher than 4
    # !: Actually this is a "TODO" according to pytorch `fold`` docs.
    raise NotImplementedError(
        "Blocking multiple dimensional weight (x.ndim >=2) is not supported."
    )
    assert x.ndim >= 2
    x_shape = [i for i in x.shape]
    block_shape = _infer_block_shape(x_shape, block_shape)
    pad_diff = _infer_padding_shape(x_shape, block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)

    padded_x = padded_x.unsqueeze(0).unsqueeze(0)
    # [1, 1, weight_shape_1, weight_shape_2, ..., weight_shape_n] -> [1, block_size_1 * block_size_2 * ... * block_size_n, num_blocks]
    blocked_x = F.unfold(
        padded_x, kernel_size=block_shape, dilation=1, padding=0, stride=block_shape
    )

    # [1, block_size_1 * block_size_2 * ... * block_size_n, num_blocks] -> [block_size_1 * block_size_2 * ... * block_size_n , num_blocks]
    blocked_x = blocked_x.squeeze(0)
    per_block_max = torch.abs(blocked_x).max(dim=0, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_multi_dim_weight(
    blocked_x: Tensor, x_shape_before_blocking, padded_x_shape, block_shape
):
    """
    [block_size_1 * block_size_2 * ... * block_size_n, num_blocks] -> [weight_shape_1, weight_shape_2, ..., weight_shape_n]
    """
    raise NotImplementedError(
        "Unblocking to multiple dimensional weight (x.ndim >=2) is not supported."
    )
    # [block_size_1 * block_size_2 * ... * block_size_n, num_blocks] -> [1, padded_x_shape[2], padded_x_shape[2], ..., padded_x_shape[n]]
    x = F.fold(
        blocked_x,
        output_size=padded_x_shape,
        kernel_size=block_shape,
        dilation=1,
        padding=0,
        stride=block_shape,
    )

    x = x.squeeze(0)
    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    x = x[indexes]
    print(f"indexes: {indexes}")
    return x
