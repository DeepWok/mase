import itertools
import numpy as np
import torch.nn as nn

from chop.tools import get_logger

from .common import SpmdShard, VALID_2D_TENSOR_SHARDINGS
from .alpa_cost_modelling import get_communication_cost

logger = get_logger(__name__)

def get_valid_linear_shardings(node_meta, mesh):
    """
    Return every valid combination of shardings for the input tensors. For an operator
    sharding to be valid, the inner dimension must have the same sharding.
    E.g. ((R, S_0), (S_0, R)) are valid, but ((R, S_0), (S_1, R)) is not.
    """
    input_shardings, output_shardings = [], []
    compute_cost_vector, communication_cost_vector = [], []

    permutations = list(itertools.product(VALID_2D_TENSOR_SHARDINGS, repeat=2))
    for p in permutations:
        output_sharding = (p[0][0], p[1][1])
        if p != ((SpmdShard.R, SpmdShard.R), (SpmdShard.R, SpmdShard.R)) and p[0][1] == p[1][0] and output_sharding in VALID_2D_TENSOR_SHARDINGS:
            input_shardings.append(p)
            output_shardings.append(output_sharding)

            compute_cost_vector.append(0)

            cost = get_communication_cost(p, node_meta["mase"], mesh)
            communication_cost_vector.append(cost)

    logger.debug(f"Valid shardings for linear layer:")
    for i, in_shard in enumerate(input_shardings):
        logger.debug(f"Sharding {i}: {in_shard} -> {output_shardings[i]}")

    return (
        input_shardings,
        output_shardings,
        np.array(compute_cost_vector),
        np.array(communication_cost_vector),
    )


ALPA_LAYERS = {
    nn.Linear: get_valid_linear_shardings,
}
