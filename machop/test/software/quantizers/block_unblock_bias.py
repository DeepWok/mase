import os
import sys

import colorlog
import torch

sys.path.append("../../")
sys.path.append("../../../software")

from logger import getLogger
from machop.modify.quantizers.utils import _block_1d_bias, _unblock_to_1d_bias

logger = getLogger("Test block and unblock bias")
bias_11 = torch.arange(11).view(-1)

test_cases = [[bias_11, [3]], [bias_11, [1]], [bias_11, [-1]], [bias_11, [12]]]


def block_and_unblock(bias, block_shape):
    logger.info(f"_block_1d_bias, block size {block_shape}")
    # bias = torch.arange(11).view(-1)
    # block_shape = [3]
    (
        blocked_bias,
        per_block_max,
        padded_bias_shape,
        inferred_block_shape,
    ) = _block_1d_bias(bias, block_shape=block_shape)
    print(
        f"""
bias: {bias},
bias_shape: {bias.shape},
block_shape: {block_shape}
inferred_block_shape: {inferred_block_shape}
blocked_bias: \n{blocked_bias},
blocked_bias.shape: {blocked_bias.shape}
per_block_max: \n{per_block_max},
per_block_max.shape: {per_block_max.shape}
        """
    )

    logger.info(f"_unblock_to_1d_bias, block size {block_shape}")
    bias_shape_before_blocking = [shape_i for shape_i in bias.shape]
    unblocked_bias = _unblock_to_1d_bias(
        blocked_x=bias, x_shape_before_blocking=bias_shape_before_blocking
    )
    print(
        f"""
unblocked_bias: {unblocked_bias},
unblocked_bias_shape: {unblocked_bias.shape}
        """
    )
    print("-" * 30)


for i, args in enumerate(test_cases):
    block_and_unblock(*args)


# logger.warning("-" * 10)
# logger.info("_block_1d_bias, block size = 1 (no blocking)")

# bias = torch.arange(11).view(-1)
# block_shape = [1]
# blocked_bias, per_block_max, padded_bias_shape, inferred_block_shape = _block_1d_bias(
#     bias, block_shape=block_shape
# )
# print(
#     f"""
# bias: {bias},
# bias_shape: {bias.shape},
# block_shape: {block_shape}
# inferred_block_shape: {inferred_block_shape}
# blocked_bias: \n{blocked_bias},
# blocked_bias.shape: {blocked_bias.shape}
# per_block_max: \n{per_block_max},
# per_block_max.shape: {per_block_max.shape}
#     """
# )
# logger.info("_unblock_to_1d_bias, block_ size = 1 (no blocking) ")
# bias_shape_before_blocking = [shape_i for shape_i in bias.shape]
# unblocked_bias = _unblock_to_1d_bias(
#     blocked_x=bias, x_shape_before_blocking=bias_shape_before_blocking
# )
# print(
#     f"""
# unblocked_bias: {unblocked_bias},
# unblocked_bias_shape: {unblocked_bias.shape}
#     """
# )
# logger.warning("-" * 10)

# logger.info("_block_1d_bias, block_size_i > x_size_i")
# bias = torch.arange(11).view(-1)
# block_shape = [12]
# blocked_bias, per_block_max, padded_bias_shape, inferred_block_shape = _block_1d_bias(
#     bias, block_shape=block_shape
# )

# print(
#     f"""
# bias: {bias},
# bias_shape: {bias.shape},
# block_shape: {block_shape}
# inferred_block_shape: {inferred_block_shape}
# blocked_bias: \n{blocked_bias},
# blocked_bias.shape: {blocked_bias.shape}
# per_block_max: \n{per_block_max},
# per_block_max.shape: {per_block_max.shape}
#     """
# )
# logger.info("_unblock_to_1d_bias, block size > x_size_i ")
# bias_shape_before_blocking = [shape_i for shape_i in bias.shape]
# unblocked_bias = _unblock_to_1d_bias(
#     blocked_x=bias, x_shape_before_blocking=bias_shape_before_blocking
# )
# print(
#     f"""
# unblocked_bias: {unblocked_bias},
# unblocked_bias_shape: {unblocked_bias.shape}
#     """
# )
# logger.warning("-" * 10)
# logger.info("_block_1d_bias, block_size_i = -1")
# bias = torch.arange(11).view(-1)
# block_shape = [-1]
# blocked_bias, per_block_max, padded_bias_shape, inferred_block_shape = _block_1d_bias(
#     bias, block_shape=block_shape
# )

# print(
#     f"""
# bias: {bias},
# bias_shape: {bias.shape},
# block_shape: {block_shape}
# inferred_block_shape: {inferred_block_shape}
# blocked_bias: \n{blocked_bias},
# blocked_bias.shape: {blocked_bias.shape}
# per_block_max: \n{per_block_max},
# per_block_max.shape: {per_block_max.shape}
#     """
# )
# logger.info("_unblock_to_1d_bias, block size = -1 ")
# bias_shape_before_blocking = [shape_i for shape_i in bias.shape]
# unblocked_bias = _unblock_to_1d_bias(
#     blocked_x=bias, x_shape_before_blocking=bias_shape_before_blocking
# )
# print(
#     f"""
# unblocked_bias: {unblocked_bias},
# unblocked_bias_shape: {unblocked_bias.shape}
#     """
# )
# print("done")
