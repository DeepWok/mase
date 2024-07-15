# Adapted from Pytorch Distributed DTensor API.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/ops/math_ops.py

from typing import cast, List, Optional, Sequence, Tuple, Union

import torch
from torch.distributed._tensor._op_schema import (
    OpStrategy,
    PlacementStrategy,
)
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    normalize_dim,
    normalize_to_torch_size,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)


def _replicate_dims_start_at(
    placements: Sequence[Placement], start_dim: int = 0
) -> Tuple[Placement, ...]:
    new_placements: List[Placement] = []
    for p in placements:
        if p.is_partial() or (isinstance(p, Shard) and p.dim >= start_dim):
            new_placements.append(Replicate())  # make it replicate
        else:
            new_placements.append(p)  # keep the placement
    return tuple(new_placements)


def replicate_reduction_dims(
    placements: Tuple[Placement, ...], reduction_dims: List[int]
) -> Tuple[Placement, ...]:
    # replicate the reduction dims if not reduction_linear
    new_placements: List[Placement] = []

    for p in placements:
        if p.is_partial():
            new_placements.append(Replicate())
        elif isinstance(p, Shard) and p.dim in reduction_dims:
            new_placements.append(Replicate())
        else:
            new_placements.append(p)

    return tuple(new_placements)


def softmax_strategy(meta, mesh):
    parent_node = meta.node.args[0]
    input_strategy = parent_node.meta["mase"]["software"]["autosharding"]["op_strategy"]
    ndim = len(meta["common"]["args"]["data_in_0"]["shape"])

    softmax_dim = meta["common"]["args"]["dim"]

    input_strategy = cast(OpStrategy, input_strategy)
    softmax_dim = cast(int, softmax_dim)
    softmax_dim = normalize_dim(softmax_dim, ndim)

    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec

        # make sure input is replicated along the softmax dim
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(
                input_src_spec.placements, [softmax_dim]
            ),
            tensor_meta=input_src_spec.tensor_meta,
        )
        # redistribute_costs.append(
        #     generate_redistribute_costs(input_strategy, input_target_spec)
        # )
        output_target_spec = input_target_spec
        output_strategy.strategies.append(
            PlacementStrategy(
                output_specs=output_target_spec,
                input_specs=[input_target_spec],
                # redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


def layer_norm_strategy(meta, mesh):

    # args must be: input, normalized_shape, weight, bias, eps
    # for None weight and bias, their corresponding objects will
    # be None as well. layer_norm_strategy returns one OpStrategy
    # for the triple return values (out, mean, rstd).
    assert len(meta["common"]["args"].keys()) == 5

    input_strategy = meta.node.args[0].meta["mase"]["software"]["autosharding"][
        "op_strategy"
    ]
    normalized_shape = meta["common"]["args"]["normalized_shape"]
    weight_strategy = meta.node.kwargs["weight"].meta["mase"]["software"][
        "autosharding"
    ]["op_strategy"]
    bias_strategy = meta.node.kwargs["bias"].meta["mase"]["software"]["autosharding"][
        "op_strategy"
    ]

    # the current layer norm implementation requires that all
    # input DTensor's sharding must be in form of OpStrategy
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)

    input_ndim = len(meta["common"]["args"]["data_in_0"]["shape"])
    axis = input_ndim - len(normalized_size)

    # we use OpStrategy because the output (out, mean, rstd)
    # should have the same placements
    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec

        # for the input tensor, we replicate it on the inner dims if necessary
        # TODO: we can avoid forcing the redistribution once we figure out
        # how to decompose layer norm
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_target_spec)
        # redistribute_costs.append(
        #     generate_redistribute_costs(input_strategy, input_target_spec)
        # )

        if weight_strategy is not None:
            assert isinstance(weight_strategy, OpStrategy)
            # patching: weight and bias sharding strategy is currently always replicate
            # So just take strategy at index 0
            # TO DO: when sharding decomposed layer norm, cross product weight strategies
            # with input/bias strategies for final OpStrategy
            weight_src_spec = weight_strategy.strategies[0].output_spec

            # for the weight tensor, we replicate it on all dims if necessary
            # TODO: we can avoid forcing the redistribution once we figure out
            # how to decompose layer norm
            weight_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_target_spec)
            # redistribute_costs.append(
            #     generate_redistribute_costs(weight_strategy, weight_target_spec)
            # )

        if bias_strategy is not None:
            assert isinstance(bias_strategy, OpStrategy)
            bias_src_spec = bias_strategy.strategies[0].output_spec

            # for the bias tensor, we replicate it on all dims if necessary
            # TODO: we can avoid forcing the redistribution once we figure out
            # how to decompose layer norm
            bias_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(bias_src_spec.placements),
                tensor_meta=bias_src_spec.tensor_meta,
            )
            op_args_target_specs.append(bias_target_spec)
            # redistribute_costs.append(
            #     generate_redistribute_costs(bias_strategy, bias_target_spec)
            # )

        # the output spec is the same as input spec
        output_target_spec = input_target_spec
        output_strategy.strategies.append(
            PlacementStrategy(
                output_specs=output_target_spec,
                input_specs=op_args_target_specs,
                # redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy
