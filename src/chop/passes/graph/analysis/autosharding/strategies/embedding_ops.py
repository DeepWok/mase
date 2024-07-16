# Adapted from Pytorch Distributed DTensor API.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/ops/embedding_ops.py

from dataclasses import dataclass, field
from typing import cast, List, Optional
import itertools

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor._op_schema import (
    OpSchema,
    OpStrategy,
    StrategyType,
    DTensorSpec,
    PlacementStrategy,
)
from torch.distributed._tensor.ops.utils import (
    is_tensor_shardable,
    generate_redistribute_costs,
)
from torch.distributed._tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)
from torch.distributed.device_mesh import DeviceMesh


aten = torch.ops.aten


@dataclass
class MaskBuffer:
    data: Optional[torch.Tensor] = None

    def materialize_mask(self, mask):
        if self.data is not None:
            raise RuntimeError("MaskBuffer has already been materialized")
        self.data = mask

    def release_mask(self):
        # TODO: evaluate if we need to release the mask buffer or the buffer
        # can just have the same lifetime as the Partial placement
        if self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")
        self.data = None

    def apply_mask(self, tensor):
        if self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")

        # NOTE: _MaskPartial is being used by the embedding op and the gather op.
        # For gather, the mask has the same dimension as the output tensor, whereas
        # the output of the embedding op has an additional dimension compare to the input,
        # hence the output masking logic below having two different cases.
        if tensor.ndim == self.data.ndim:
            tensor[self.data] = 0.0
        else:
            tensor[self.data, :] = 0.0


@dataclass(frozen=True)
class _MaskPartial(Partial):
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """

    logical_dim_size: int = -1
    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # override parent logic to perform partial mask for embedding
        num_chunks = mesh.size(mesh_dim)
        # get local shard size and offset on the embedding_dim
        local_shard_size, local_offset_on_dim = Shard._local_shard_size_on_dim(
            self.logical_dim_size,
            num_chunks,
            mesh.get_local_rank(mesh_dim),
            return_offset=True,
        )
        # Build the input mask and save it for the current partial placement
        # this is so that the output of embedding op can reuse the same partial
        # placement saved mask to perform mask + reduction
        mask = (tensor < local_offset_on_dim) | (
            tensor >= local_offset_on_dim + local_shard_size
        )
        # mask the input tensor
        masked_tensor = tensor.clone() - local_offset_on_dim
        masked_tensor[mask] = 0
        # materialize the mask buffer to be used for reduction
        self.mask_buffer.materialize_mask(mask)
        return masked_tensor

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # by the time we ned reduction, we should have already saved the mask
        assert self.mask_buffer.data is not None

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # perform sum reduction
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # by the time we ned reduction, we should have already saved the mask
        assert self.mask_buffer.data is not None

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # call reduce_shard_tensor of the shard_spec.
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaskPartial):
            return False

        # if either data is not None, we invalidate the sharding cache, as this indicates
        # the current MaskPartial placement is still in use and should not be used for cache hit.
        if self.mask_buffer.data is not None or other.mask_buffer.data is not None:
            return False

        return (
            self.reduce_op == other.reduce_op
            and self.logical_dim_size == other.logical_dim_size
        )

    def __hash__(self) -> int:
        return 1 + hash(
            (self.logical_dim_size, id(self.mask_buffer.data), self.reduce_op)
        )

    def __repr__(self) -> str:
        """
        machine readable representation of the MaskPartial placement
        """
        return f"_MaskPartial(logical_dim_size={self.logical_dim_size})"

    def __str__(self) -> str:
        """
        human readable representation of the MaskPartial placement
        """
        return "MaskP"


def expand_to_full_mesh_op_strategy(
    meta,
    mesh: DeviceMesh,
    single_mesh_dim_strategies: List[List[Placement]],
    *,
    input_index: int = 1,
    inplace_op: bool = False,
) -> OpStrategy:
    # Expand the single_mesh_dim_strategies to full mesh dim strategies.
    all_mesh_dim_strategies = [single_mesh_dim_strategies] * mesh.ndim

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(
                DTensorSpec(
                    mesh,
                    tuple(specs),
                    tensor_meta=TensorMeta(
                        shape=meta["common"]["results"]["data_out_0"]["shape"],
                        stride=None,
                        dtype=meta["common"]["results"]["data_out_0"]["torch_dtype"],
                    ),
                )
            )

        input_specs = spec_list[input_index:]
        # input_args_strategy = op_schema.args_strategy
        input_args_strategy = tuple(
            arg.meta["mase"]["software"]["autosharding"]["op_strategy"]
            for arg in meta.node.args
        )
        assert len(input_specs) == len(input_args_strategy)
        self_spec = input_args_strategy[0].strategies[0].output_spec
        if inplace_op and self_spec.placements != input_specs[0].placements:
            # if it's inplace op, we would only allow the placement strategy to be added when the
            # input_spec matches the first argument's runtime sharding, otherwise we skip
            continue

        # check inputs shardable
        inputs_shardable = all(
            is_tensor_shardable(inp.shape, s)
            for inp, s in zip(input_args_strategy, input_specs)
        )

        # only add to the all_strategies list when all inputs are shardable
        if inputs_shardable:
            redistribute_cost = [
                generate_redistribute_costs(input_strategy, input_spec)
                for input_strategy, input_spec in zip(input_args_strategy, input_specs)
            ]
            strategy = PlacementStrategy(
                output_specs=(
                    tuple(spec_list[:input_index]) if input_index > 1 else spec_list[0]
                ),
                input_specs=input_specs,
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strategy)

    return OpStrategy(all_strategies)


def embedding_strategy(meta, mesh) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    """
    weight_shape = meta["common"]["args"]["data_in_0"]["shape"]
    indices_shape = meta["common"]["args"]["data_in_1"]["shape"]
    output_emd_dim = len(indices_shape)

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, weight, input_indices]
    # first we always have replicate all for inputs and output
    all_replicate: List[Placement] = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # colwise sharding, output shard on last dim, weight shard on dim 1, input replicate
    colwise_sharding = [Shard(output_emd_dim), Shard(1), Replicate()]
    single_mesh_dim_strategies.append(colwise_sharding)

    # rowwise sharding, output is embedding partial, weight shard on dim 0, input accepts embedding partial
    embedding_partial_placement = _MaskPartial(logical_dim_size=weight_shape[0])

    # NOTE we want to reuse the same mask partial placement so that we can reuse the same mask that generates
    # from the input indices and use it for output reduction
    rowwise_sharding = [
        embedding_partial_placement,
        Shard(0),
        embedding_partial_placement,
    ]
    single_mesh_dim_strategies.append(rowwise_sharding)

    # batch dim sharding, weight replicated, input can shard on any dim, output follows input
    for input_dim in range(len(indices_shape)):
        batch_sharding = [Shard(input_dim), Replicate(), Shard(input_dim)]
        single_mesh_dim_strategies.append(batch_sharding)

    return expand_to_full_mesh_op_strategy(meta, mesh, single_mesh_dim_strategies)
