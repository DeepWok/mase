from enum import Enum
import torch.nn as nn
import itertools
import random


class Shard(Enum):
    R = 1
    S_0 = 2
    S_1 = 3
    S_01 = 4

    def __repr__(self):
        return self.name


VALID_2D_TENSOR_SHARDINGS = [
    (Shard.R, Shard.R),
    (Shard.R, Shard.S_0),
    (Shard.R, Shard.S_1),
    (Shard.R, Shard.S_01),
    (Shard.S_0, Shard.R),
    (Shard.S_0, Shard.S_1),
    (Shard.S_1, Shard.R),
    (Shard.S_1, Shard.S_0),
    (Shard.S_01, Shard.R),
]


def get_valid_linear_shardings():
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
        if p[0][1] == p[1][0] and output_sharding in VALID_2D_TENSOR_SHARDINGS:
            input_shardings.append(p)
            output_shardings.append(output_sharding)

            compute_cost_vector.append(random.random())

            # TO DO: derive communication cost from the sharding
            communication_cost_vector.append(random.random())

    for i, in_shard in enumerate(input_shardings):
        print(f"Sharding {i}: {in_shard} -> {output_shardings[i]}")

    return (
        input_shardings,
        output_shardings,
        compute_cost_vector,
        communication_cost_vector,
    )


SHARDING_ALGOS = {
    nn.Linear: get_valid_linear_shardings,
}
