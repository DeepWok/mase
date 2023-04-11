import os
import sys

import colorlog
import torch

sys.path.append("../../")
sys.path.append("../../../software")

from logger import getLogger
from machop.modify.quantizers.utils import _block_2d_weight, _unblock_to_2d_weight

logger = getLogger("Test block and unblock 2D weight")

weight_4x6 = torch.arange(4 * 6, dtype=torch.float).reshape(4, 6)
test_cases = [
    [weight_4x6, [2, 3]],
    [weight_4x6, [3, 4]],
    [weight_4x6, [1, 4]],
    [weight_4x6, [3, 1]],
    [weight_4x6, [-1, 4]],
    [weight_4x6, [3, -1]],
    [weight_4x6, [4]],
    [weight_4x6, [-1]],
]


def test_block_and_unblock(weight, block_shape):
    logger.info(f"_block_2d_weight, block shape = {block_shape}")

    # weight = torch.arange(4 * 6, dtype=torch.float).reshape(4, 6)
    # block_shape = [2, 3]

    (
        blocked_weight,
        per_block_max,
        padded_weight_shape,
        inferred_block_shape,
    ) = _block_2d_weight(weight, block_shape=block_shape)

    print(
        f"""
weight: \n {weight},
weight_shape: {weight.shape},
block_shape: {block_shape}
inferred_block_shape: {inferred_block_shape}
blocked_weight[:, [0]] (0th block):\n {blocked_weight[:, [0]]}
blocked_weight[:, [0]].shape:\n {blocked_weight[:, [0]].shape}
blocked_weight: \n {blocked_weight},
blocked_weight.shape: {blocked_weight.shape}
per_block_max: \n {per_block_max},
per_block_max.shape: {per_block_max.shape}
        """
    )

    logger.info(f"_unblock_2d_weight, block shape {block_shape}")
    weight_shape_before_blocking = [i for i in weight.shape]
    unblocked_weight = _unblock_to_2d_weight(
        blocked_weight,
        weight_shape_before_blocking,
        padded_x_shape=padded_weight_shape,
        block_shape=inferred_block_shape,
    )
    print(
        f"""
unblocked_bias:\n {unblocked_weight},
unblocked_bias_shape:\n {unblocked_weight.shape}
        """
    )

    print("-" * 30)


for i, args in enumerate(test_cases):
    logger.info(f"{i}-th test case")
    test_block_and_unblock(*args)

logger.info("All test cases passed")
