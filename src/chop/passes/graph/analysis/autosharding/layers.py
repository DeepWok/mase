import itertools
import operator

import torch
import torch.nn.functional as F
from torch.distributed._tensor._op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import (
    Replicate,
    Shard,
    DTensorSpec,
    TensorMeta,
)

from chop.tools import get_logger

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
from .ops.embedding_ops import embedding_strategy

logger = get_logger(__name__)


def placeholder_or_getattr_strategy(meta, mesh, skip_fully_replicated=False):
    ndims = len(meta["common"]["results"]["data_out_0"]["shape"])
    opts = [Replicate()] + [Shard(dim) for dim in range(ndims)]

    tensor_meta = TensorMeta(
        shape=meta["common"]["results"]["data_out_0"]["shape"],
        stride=None,
        dtype=meta["common"]["results"]["data_out_0"]["torch_dtype"],
    )

    shardings = []
    for sharding in itertools.product(opts, repeat=2):
        if skip_fully_replicated and sharding == (Replicate(), Replicate()):
            continue
        spec = DTensorSpec(mesh=mesh, placements=sharding, tensor_meta=tensor_meta)
        shardings.append(PlacementStrategy(input_specs=spec, output_specs=spec))
    return OpStrategy(shardings)


def fully_replicated_strategy(meta, mesh):
    """
    Output of ops like size, getitem etc are always fully replicated
    """
    sharding = [Replicate(), Replicate()]

    # call_method nodes don't list input tensor in the args list, but
    # tensor is copied into meta["common"]["self"] when add_value = True
    # is passed to add_common_metadata_pass
    if meta.node.op == "call_method":
        in_shape = meta["common"]["self"].shape
        in_dtype = meta["common"]["self"].dtype
    else:
        first_arg_key = (
            "data_in_0"
            if "data_in_0" in meta["common"]["args"]
            else [i for i in meta["common"]["args"].keys()][0]
        )
        arg = meta["common"]["args"][first_arg_key]
        if isinstance(arg, dict):
            in_shape = arg["shape"]
            in_dtype = arg["torch_dtype"]
        else:
            arg = torch.Tensor(arg)
            in_shape = arg.shape
            in_dtype = arg.dtype

    in_spec = DTensorSpec(
        mesh,
        sharding,
        tensor_meta=TensorMeta(shape=in_shape, stride=None, dtype=in_dtype),
    )

    dtype_key = (
        "torch_dtype"
        if "torch_dtype" in meta["common"]["results"]["data_out_0"].keys()
        else "type"
    )
    out_dtype = meta["common"]["results"]["data_out_0"][dtype_key]
    out_spec = DTensorSpec(
        mesh,
        sharding,
        tensor_meta=TensorMeta(
            shape=meta["common"]["results"]["data_out_0"]["shape"],
            stride=None,
            dtype=out_dtype,
        ),
    )

    shardings = [PlacementStrategy(input_specs=in_spec, output_specs=out_spec)]

    return OpStrategy(shardings)


AUTOSHARDING_FUNCTIONS = {
    torch.transpose: transpose_strategy,
    torch.mm: mm_strategy,
    torch.addmm: addmm_strategy,
    torch.bmm: bmm_strategy,
    torch.baddbmm: baddmm_strategy,
    torch.add: linear_pointwise_strategy,
    operator.add: linear_pointwise_strategy,
    operator.truediv: pointwise_strategy,
    F.gelu: pointwise_strategy,
    torch.sub: pointwise_strategy,
    torch.gt: pointwise_strategy,
    operator.gt: pointwise_strategy,
    operator.sub: pointwise_strategy,
    torch.matmul: bmm_strategy,
    torch.softmax: softmax_strategy,
    F.softmax: softmax_strategy,
    F.layer_norm: layer_norm_strategy,
    torch.ones: fully_replicated_strategy,
    torch.full: fully_replicated_strategy,
    getattr: fully_replicated_strategy,
    F.embedding: embedding_strategy,
}

AUTOSHARDING_METHODS = {
    "view": get_reshape_strategy(torch.Tensor.view),
    "reshape": get_reshape_strategy(torch.Tensor.reshape),
    "expand": get_reshape_strategy(torch.Tensor.expand),
    "permute": get_reshape_strategy(torch.permute),
    "transpose": get_reshape_strategy(torch.transpose),
}

IMPLICIT_FUNCS = [operator.getitem]

IMPLICIT_METHODS = ["size"]
