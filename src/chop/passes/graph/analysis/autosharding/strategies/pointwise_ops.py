# Adapted from Pytorch Distributed DTensor API.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/ops/pointwise_ops.py

from typing import List, Sequence, Tuple

import torch
from torch.distributed._tensor._op_schema import (
    _is_inplace_op,
    _is_out_variant_op,
    OpStrategy,
    PlacementStrategy,
)
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    map_placements_after_broadcast,
    normalize_dim,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)

from chop.tools import get_logger

from .common import fully_replicated_strategy

logger = get_logger(__name__)


def pointwise_strategy(meta, mesh, linearity=False):
    max_shards_strategy_index = -1
    max_shards = -1
    followed_strategy = None

    # if _is_inplace_op(op_schema.op):
    #     # inplace op should follow the first arg strategy
    #     followed_strategy = op_schema.args_schema[0]
    # elif _is_out_variant_op(op_schema.op):
    #     # out variant op should follow the out kwarg strategy
    #     followed_strategy = op_schema.kwargs_schema["out"]
    # else:

    # normal pointwise op, we choose to follow the arg with
    # the max shards in case operands needs reshard
    for idx, arg in enumerate(meta.node.args):
        if not isinstance(arg, torch.fx.Node):
            continue
        arg_strategy = arg.meta["mase"]["software"]["autosharding"]["op_strategy"]

        arg_max_shards = arg_strategy.max_num_shards()
        if arg_max_shards > max_shards:
            max_shards_strategy_index = idx
            max_shards = arg_max_shards
            followed_strategy = arg_strategy

    assert isinstance(followed_strategy, OpStrategy), f"no strategy to follow!"

    return common_pointwise_strategy(
        meta, mesh, followed_strategy, linearity, max_shards_strategy_index
    )


def common_pointwise_strategy(
    meta, mesh, followed_strategy, linearity, followed_strategy_index=0
):
    # handle broadcasting
    parsed_args = []
    for arg in meta["common"]["args"].values():
        if isinstance(arg, dict):
            parsed_args.append(torch.zeros(arg["shape"]))
        elif isinstance(arg, torch.Size):
            parsed_args.append(torch.Tensor(list(arg)))
        elif isinstance(arg, (tuple, list)):
            parsed_args.append(torch.Tensor(arg))
        elif isinstance(arg, torch.Tensor):
            parsed_args.append(arg)
        elif isinstance(arg, (float, int)):
            parsed_args.append(torch.Tensor([arg]))
        else:
            logger.warning(
                f"Unrecognized arg type: {type(arg)}, defaulting to fully replicated strategy."
            )
            return fully_replicated_strategy(meta, mesh)

    common_shape = torch.broadcast_shapes(*[arg.shape for arg in parsed_args])

    # Extract followed argument shape
    followed_shape = parsed_args[followed_strategy_index].shape

    # Iterate through followed argument's strategies to cast output shardings
    pointwise_strategy = OpStrategy([])
    for placement_strategy in followed_strategy.strategies:
        spec_to_follow = placement_strategy.output_spec
        out_placements: List[Placement] = []
        for placement in spec_to_follow.placements:
            if isinstance(placement, Shard):
                shard_dim = normalize_dim(placement.dim, len(followed_shape))
                common_ndim = len(common_shape)
                new_shard_dim = common_ndim - len(followed_shape) + shard_dim
                out_placements.append(Shard(new_shard_dim))
            elif isinstance(placement, Partial) and not linearity:
                # clear the partial placemnet if op does not support linearity
                # by default we just replicate the partial, need to see if this
                # is optimal for all cases
                out_placements.append(Replicate())
            else:
                out_placements.append(placement)

        input_specs: List[DTensorSpec] = []
        # redistribute_costs: List[List[float]] = []
        for arg_node in meta.node.args:
            if not isinstance(arg_node, torch.fx.Node):
                continue
            input_arg = arg_node.meta["mase"]["software"]["autosharding"]["op_strategy"]
            if isinstance(input_arg, OpStrategy):
                # every arg follow the out_placements, but need to handle broadcasting
                input_arg_spec = input_arg.strategies[0].output_spec
                input_arg_dims_map = infer_broadcast_dims_map(
                    common_shape,
                    arg_node.meta["mase"]["common"]["results"]["data_out_0"]["shape"],
                )
                input_target_placements = map_placements_after_broadcast(
                    tuple(out_placements),
                    common_shape,
                    input_arg_dims_map,
                )
                input_arg_target_spec = DTensorSpec(
                    mesh=mesh,
                    placements=input_target_placements,
                    tensor_meta=input_arg_spec.tensor_meta,
                )
                input_specs.append(input_arg_target_spec)
                # redistribute_costs.append(
                #     generate_redistribute_costs(input_arg, input_arg_target_spec)
                # )

        dtype = meta["common"]["results"]["data_out_0"].get(
            "torch_dtype", torch.float32
        )
        pointwise_strategy.strategies.append(
            PlacementStrategy(
                output_specs=DTensorSpec(
                    mesh=mesh,
                    placements=tuple(out_placements),
                    tensor_meta=TensorMeta(
                        shape=meta["common"]["results"]["data_out_0"]["shape"],
                        stride=None,
                        dtype=dtype,
                    ),
                ),
                input_specs=input_specs,
                # redistribute_cost=redistribute_costs,
            )
        )
    return pointwise_strategy


def linear_pointwise_strategy(meta, mesh):
    """
    Linear pointwise operators can propagate pending reductions.
    For example, c = add(a, b); if a is pending sum, then c will be
    pending sum as well without any communication overhead.
    """
    return pointwise_strategy(meta, mesh, linearity=True)
