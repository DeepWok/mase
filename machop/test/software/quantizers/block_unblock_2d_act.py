import os
import sys

import colorlog
import torch

sys.path.append("../../")
sys.path.append("../../../software")

from logger import getLogger
from machop.modify.quantizers.utils import (
    _block_2d_activation,
    _unblock_to_2d_activation,
)

logger = getLogger("Test block and unblock 2D activation")


def test_block_and_unblock(act, block_shape):
    logger.info(f"_block_2d_activation, block size: {block_shape}")

    (
        blocked_act,
        per_block_max,
        padded_act_shape,
        inferred_block_shape,
    ) = _block_2d_activation(act, block_shape=block_shape)

    print(
        f"""
act: \n {act},
act_shape: {act.shape},
block_shape: {block_shape}
inferred_block_shape: {inferred_block_shape}
blocked_act: \n {blocked_act},
blocked_act.shape: {blocked_act.shape}
per_block_max: \n {per_block_max},
per_block_max.shape: {per_block_max.shape}
        """
    )

    logger.info(f"_unblock_to_2d_activation, block size: {block_shape}")
    act_shape_before_blocking = [i for i in act.shape]
    unblocked_act = _unblock_to_2d_activation(blocked_act, act_shape_before_blocking)
    print(
        f"""
unblocked_bias:\n {unblocked_act},
unblocked_bias_shape:\n {unblocked_act.shape}
        """
    )
    print("-" * 30)


act_2x12 = torch.arange(2 * 12).reshape(2, 12)

test_cases = [[act_2x12, [7]], [act_2x12, [13]], [act_2x12, [1]], [act_2x12, [-1]]]

for i, args in enumerate(test_cases):
    logger.info(f"Test case {i}")
    test_block_and_unblock(*args)
