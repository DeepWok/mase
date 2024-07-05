import itertools
import numpy as np
import torch.nn as nn

from chop.tools import get_logger
from chop.models.patched.bert.modeling_bert import BertSelfAttention

from .common import SpmdShard, VALID_2D_TENSOR_SHARDINGS
from .alpa_cost_modelling import get_communication_cost


logger = get_logger(__name__)


def is_valid_2d_sharding(sharding):
    if len(sharding) > 2:
        return sharding[1:] in VALID_2D_TENSOR_SHARDINGS
    else:
        return sharding in VALID_2D_TENSOR_SHARDINGS


def is_valid_sharding_pair(sharding_pair):
    return sharding_pair[0][-1] == sharding_pair[1][-2]


def is_fully_replicated(sharding_pair):
    return all(all(dimp == SpmdShard.R for dimp in subp) for subp in sharding_pair)


def get_valid_2d_shardings(node_meta, mesh, module):
    """
    Return every valid combination of shardings for the input tensors. For an operator
    sharding to be valid, the inner dimension must have the same sharding.
    E.g. ((R, S_0), (S_0, R)) are valid, but ((R, S_0), (S_1, R)) is not.
    """
    input_shardings = []
    output_shardings = []
    compute_cost_vector = []
    communication_cost_vector = []

    out_rank = len(node_meta["mase"]["common"]["results"]["data_out_0"]["shape"])

    for perm in itertools.product(VALID_2D_TENSOR_SHARDINGS, repeat=2):
        if out_rank > 2:
            perm = tuple((SpmdShard.R,) * (out_rank - 2) + p for p in perm)
        output_sharding = tuple(
            (SpmdShard.R,) * (out_rank - 2) + (perm[0][-2], perm[1][-1])
        )
        if (
            not is_fully_replicated(perm)
            and is_valid_sharding_pair(perm)
            and is_valid_2d_sharding(output_sharding)
        ):
            input_shardings.append({"data_in_0": perm[0], "weight": perm[1]})
            output_shardings.append(output_sharding)

            compute_cost_vector.append(0)
            communication_cost_vector.append(
                get_communication_cost(perm, node_meta["mase"], mesh)
            )

    return (
        input_shardings,
        output_shardings,
        np.array(compute_cost_vector),
        np.array(communication_cost_vector),
    )


def get_valid_linear_shardings(node_meta, mesh, module):
    return get_valid_2d_shardings(node_meta, mesh, module)


def get_valid_layernorm_shardings(node_meta, mesh, module):
    rank = len(node_meta["mase"]["common"]["results"]["data_out_0"]["shape"])
    valid_input_shardings = [{"data_in_0": (SpmdShard.R,) * rank}]
    valid_output_shardings = [(SpmdShard.R,) * rank]
    compute_cost_vector = [0]
    communication_cost_vector = [0]
    return (
        valid_input_shardings,
        valid_output_shardings,
        np.array(compute_cost_vector),
        np.array(communication_cost_vector),
    )


def get_valid_embedding_shardings(node_meta, mesh, module):
    weight_rank = len(module.weight.shape)
    data_in_rank = len(node_meta["mase"]["common"]["args"]["data_in_0"]["shape"])
    valid_input_shardings = [
        {
            "data_in_0": (SpmdShard.R,) * data_in_rank,
            "weight": (SpmdShard.R,) * weight_rank,
        }
    ]
    valid_output_shardings = [(SpmdShard.R,) * data_in_rank]
    compute_cost_vector = [0]
    communication_cost_vector = [0]
    return (
        valid_input_shardings,
        valid_output_shardings,
        np.array(compute_cost_vector),
        np.array(communication_cost_vector),
    )


ALPA_LAYERS = {
    nn.Linear: get_valid_linear_shardings,
    nn.LayerNorm: get_valid_layernorm_shardings,
    nn.Embedding: get_valid_embedding_shardings,
}
