import torch
from torch import Tensor
import torch.nn.functional as F
import logging

from .simulation_tile import sram_tile, reram_tile, pcm_tile

logger = logging.getLogger(__name__)

# ToDo: ADD Drift Noise
# ToDo: add scaling factor, from weight to gt


def mm_tile(x: Tensor, weight: Tensor, config: dict):
    return x @ weight


def _pim_tile(x, weight, config):
    if config.get("tile_type") == "digital":
        return sram_tile(x, weight, config)
    elif config.get("tile_type") == "reram":
        return reram_tile(x, weight, config)
    elif config.get("tile_type") == "pcm":
        return pcm_tile(x, weight, config)
    elif config.get("tile_type") == "original":
        return mm_tile(x, weight, config)
    else:
        raise ValueError(f"Invalid tile type: {config.get('tile_type')}")


def pim_core(x: Tensor, weight: Tensor, config: dict):
    """
    The digital mm is conducted in the following way:
    1. Reshape the x and weight to the vector-wise
    2. Conduct the digital mm
    3. Sum the result after blocking
    4. Reshape the result to the original shape

    the config should contain the following:
    - x_quant_type
    - weight_quant_type
    - rescale_dim
    - approximate_mode
    """
    x_shape = x.shape
    weight_shape = weight.shape

    core_size = config.get("core_size", None)

    if core_size is None:
        logger.debug(f"No core size is provided, using the original mm")
        return pim_tile(x, weight, config)

    # Pre-compute padding requirements
    x_pad_size_0 = (core_size - (x_shape[-2] % core_size)) % core_size
    x_pad_size_1 = (core_size - (x_shape[-1] % core_size)) % core_size

    w_pad_size_0 = (core_size - (weight_shape[0] % core_size)) % core_size
    w_pad_size_1 = (core_size - (weight_shape[1] % core_size)) % core_size

    # Pad x if needed
    px = F.pad(x, (0, x_pad_size_1, 0, x_pad_size_0), "constant", 0)
    px_shape = px.shape

    pw = F.pad(weight, (0, w_pad_size_1, 0, w_pad_size_0), "constant", 0)
    pw_shape = pw.shape

    # in order to follow the law of torch.mm
    # px will be reshaped to (1, -1, core_size, px_row_depth, core_size)
    # and be view as (1, -1, px_row_depth, core_size, core_size)
    # pw will be reshaped to (px_row_depth, 1           , -1, core_size, core_size)
    # and be view as         (pw_col_depth, pw_row_depth,  1,            core_size, core_size)
    # the output will be                   (pw_row_depth, -1, core_size, core_size)
    # the target shape will be (-1, px_col, pw_row)
    # so then the output will be permute to (-1, core_size, core_size, pw_row_depth)
    # so then the output will be reshaped to (-1, padding_px_row, padding_pw_col)

    # the output will be reshaped to (1, -1, pw_row_depth, core_size, core_size)
    # and be view as (1, -1, pw_row_depth, core_size, core_size)

    # the output will be summed to (1, -1, pw_row_depth, core_size, core_size)
    px = px.view(1, -1, core_size, px_shape[-1] // core_size, core_size).permute(
        3, 0, 1, 2, 4
    )
    pw = pw.view(
        1, pw_shape[0] // core_size, core_size, pw_shape[1] // core_size, core_size
    ).permute(1, 3, 0, 2, 4)

    pout = pim_tile(px, pw, config)
    pout = pout.sum(dim=0).permute(1, 2, 0, 3)

    pout = pout.reshape(-1, px_shape[-2], pw_shape[-1])
    out = pout[:, : x_shape[-2], : weight_shape[1]]
    out = out.view(*x_shape[:-1], weight_shape[1])

    return out


class PIMTile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, config):
        ctx.save_for_backward(x, weight)
        ctx.config = config
        return _pim_tile(x, weight, config)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_input = grad_output @ weight.transpose(-2, -1)
        grad_weight = x.transpose(-2, -1) @ grad_output
        return grad_input, grad_weight, None


def pim_tile(x, weight, config):
    return PIMTile.apply(x, weight, config)
