import itertools

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

logger = get_logger(__name__)


def find_shape_and_dtype(arg):

    # If the argument in meta["common"]["args"][key] is correctly
    # formulated with data, just extract shape and dtype
    if isinstance(arg, dict):
        in_shape = arg["shape"]
        in_dtype = arg["torch_dtype"]

    # Otherwise, depends on the type of argument
    elif isinstance(arg, torch.Size) or isinstance(arg, (tuple, list)):
        in_shape = (len(arg),)
        in_dtype = type(arg[0])
    elif isinstance(arg, (float, int)):
        in_shape = (1,)
        in_dtype = type(arg)
    else:
        logger.warning(f"Unknown type for arg: {arg}")
        in_shape = tuple()
        in_dtype = type(arg)

    return in_shape, in_dtype


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
        in_shape, in_dtype = find_shape_and_dtype(arg)

    in_spec = DTensorSpec(
        mesh,
        sharding,
        tensor_meta=TensorMeta(
            shape=in_shape,
            stride=None,
            dtype=in_dtype,
        ),
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
