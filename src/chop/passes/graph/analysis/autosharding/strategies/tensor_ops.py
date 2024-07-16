# Adapted from Pytorch Distributed DTensor API.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/ops/tensor_ops.py

from torch.distributed._tensor._op_schema import (
    OpStrategy,
    PlacementStrategy,
    StrategyType,
)
from torch.distributed._tensor.ops.utils import (
    is_tensor_partial,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Replicate,
    TensorMeta,
)


def tensor_op_strategy(meta, mesh) -> StrategyType:
    # Default strategy by default just propagate the first input strategy
    select_strategy = meta.node.args[0].meta["mase"]["software"]["autosharding"][
        "op_strategy"
    ]
    assert isinstance(select_strategy, OpStrategy)

    node_args = list(meta["common"]["args"].keys())
    if len(node_args) > 0:
        first_arg_name = node_args[0]
        arg_shape, arg_dtype = (
            meta["common"]["args"][first_arg_name]["shape"],
            meta["common"]["args"][first_arg_name]["torch_dtype"],
        )

    else:
        arg_shape, arg_dtype = (
            meta["common"]["self"].shape,
            meta["common"]["self"].dtype,
        )

    first_result = list(meta["common"]["results"].keys())[0]
    result_shape, result_dtype = (
        meta["common"]["results"][first_result]["shape"],
        meta["common"]["results"][first_result]["torch_dtype"],
    )

    default_strategy = []
    for strategy in select_strategy.strategies:
        # we create new DTensorSpecs even for default strategy to assure that
        # the tensor metas are distinct between the arguments and outputs
        default_strategy.append(
            PlacementStrategy(
                input_specs=(
                    DTensorSpec(
                        mesh=strategy.output_spec.mesh,
                        placements=strategy.output_spec.placements,
                        tensor_meta=TensorMeta(
                            shape=arg_shape, dtype=arg_dtype, stride=None
                        ),
                    ),
                )
                * len(meta.node.args),
                output_specs=DTensorSpec(
                    mesh=strategy.output_spec.mesh,
                    placements=strategy.output_spec.placements,
                    tensor_meta=TensorMeta(
                        shape=result_shape, dtype=result_dtype, stride=None
                    ),
                ),
            )
        )
    return OpStrategy(default_strategy)


def tensor_equal_strategy(meta, mesh) -> StrategyType:
    # equal_strategy deals with ops that comparing two tensor, we need to make sure
    # sharding layout the same with two operands, we choose to follow the arg with max
    # num of shards, still keep is_same_size here for completeness as they share the
    # same strategy in theory.
    self_strategy, other_strategy = (
        meta.node.args[0].meta["mase"]["software"]["autosharding"]["op_strategy"],
        meta.node.args[1].meta["mase"]["software"]["autosharding"]["op_strategy"],
    )
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(other_strategy, OpStrategy)

    select_strategy = (
        self_strategy
        if self_strategy.max_num_shards() >= other_strategy.max_num_shards()
        else other_strategy
    )
    equal_strategy = OpStrategy([])

    for arg_strategy in select_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            # if the arg_spec have partial, reshard to replicate
            # otherwise local shard tensor comparison would be invalid
            output_spec = DTensorSpec(
                mesh=arg_spec.mesh,
                placements=tuple(
                    Replicate() if isinstance(p, Partial) else p
                    for p in arg_spec.placements
                ),
            )
            equal_strategy.strategies.append(
                PlacementStrategy(output_specs=output_spec)
            )
        else:
            equal_strategy.strategies.append(PlacementStrategy(arg_spec))
    return equal_strategy
