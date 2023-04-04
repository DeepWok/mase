import os
import sys

import colorlog
import torch

sys.path.append("../../")
sys.path.append("../../../software")

from logger import getLogger
from machop.modify.quantizers.utils import (
    _block_3d_activation,
    _unblock_to_3d_activation,
)

logger = getLogger("Test block and unblock 3D act")

act_2_4_5 = torch.arange(2 * 4 * 6, dtype=torch.float).reshape(2, 4, 6)
test_cases = [
    [act_2_4_5, [2, 3]],
    [act_2_4_5, [3, 4]],
    [act_2_4_5, [1, 4]],
    [act_2_4_5, [3, 1]],
    [act_2_4_5, [3, -1]],
    [act_2_4_5, [-1, 4]],
    [act_2_4_5, [-1]],
]


def test_block_and_unblock(act, block_shape):
    logger.info(f"_block_multi_dim_act, block shape = {block_shape}")

    # act = torch.arange(4 * 6, dtype=torch.float).reshape(4, 6)
    # block_shape = [2, 3]

    (
        blocked_act,
        per_block_max,
        padded_act_shape,
        inferred_block_shape,
    ) = _block_3d_activation(act, block_shape=block_shape)

    print(
        f"""
act: \n {act},
act_shape: {act.shape},
block_shape: {block_shape}
inferred_block_shape: {inferred_block_shape}
blocked_act[:, :, [0]] (0th block):\n {blocked_act[:,:, [0]]}
blocked_act[:, :, [0]].shape:\n {blocked_act[:,:, [0]].shape}
blocked_act: \n {blocked_act},
blocked_act.shape: {blocked_act.shape}
per_block_max: \n {per_block_max},
per_block_max.shape: {per_block_max.shape}
        """
    )

    logger.info(f"_unblock_multi_3d_act, block shape {block_shape}")
    act_shape_before_blocking = [i for i in act.shape]
    unblocked_act = _unblock_to_3d_activation(
        blocked_act,
        act_shape_before_blocking,
        padded_x_shape=padded_act_shape,
        block_shape=inferred_block_shape,
    )
    print(
        f"""
unblocked_bias:\n {unblocked_act},
unblocked_bias_shape:\n {unblocked_act.shape}
        """
    )

    print("-" * 30)


for i, args in enumerate(test_cases):
    logger.info(f"{i}-th test case")
    test_block_and_unblock(*args)

logger.info("All test cases passed")
