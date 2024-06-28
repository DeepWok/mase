import itertools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor._op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import Replicate, Shard, DTensorSpec

from chop.tools import get_logger
from chop.models.patched.bert.modeling_bert import BertSelfAttention

from .alpa_cost_modelling import get_communication_cost

from .ops.matrix_ops import (
    transpose_strategy,
    mm_strategy,
    addmm_strategy,
    bmm_strategy,
    baddmm_strategy,
)

from .ops.view_ops import get_reshape_strategy
from .ops.pointwise_ops import pointwise_strategy, linear_pointwise_strategy

from .ops.math_ops import softmax_strategy, layer_norm_strategy

logger = get_logger(__name__)

ALPA_FUNCTIONS = {
    torch.transpose: transpose_strategy,
    torch.mm: mm_strategy,
    torch.addmm: addmm_strategy,
    torch.bmm: bmm_strategy,
    torch.baddbmm: baddmm_strategy,
    torch.add: linear_pointwise_strategy,
    operator.add: linear_pointwise_strategy,
    operator.truediv: pointwise_strategy,
    F.gelu: pointwise_strategy,
    torch.matmul: bmm_strategy,
    torch.softmax: softmax_strategy,
    F.softmax: softmax_strategy,
    F.layer_norm: layer_norm_strategy
}

ALPA_METHODS = {
    "view": get_reshape_strategy(torch.Tensor.view),
    "reshape": get_reshape_strategy(torch.Tensor.reshape),
    "expand": get_reshape_strategy(torch.Tensor.expand),
    "permute": get_reshape_strategy(torch.permute),
    "transpose": get_reshape_strategy(torch.transpose)
}

IMPLICIT_FUNCS = [
    operator.getitem
]

IMPLICIT_METHODS = [
    "size"
]

def placeholder_or_getattr_strategy(meta, mesh):
    ndims = len(meta["common"]["results"]["data_out_0"]["shape"])
    opts = [Replicate()] + [Shard(dim) for dim in range(ndims)]
    shardings = []
    for sharding in itertools.product(opts, repeat=2):
        spec = DTensorSpec(mesh, sharding)
        shardings.append(PlacementStrategy(
            input_specs=spec,
            output_specs=spec
        ))
    return OpStrategy(shardings)

def fully_replicated_strategy(meta, mesh):
    """
    Output of ops like size, getitem etc are always fully replicated
    """
    sharding = [Replicate(), Replicate()]
    spec = DTensorSpec(mesh, sharding)
    shardings = [
        PlacementStrategy(
            input_specs=spec,
            output_specs=spec
        )
    ]
    return OpStrategy(shardings)