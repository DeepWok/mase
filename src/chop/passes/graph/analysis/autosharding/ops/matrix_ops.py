# Adapted from https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/ops/matrix_ops.py

import itertools
from typing import List, Optional

import torch
from torch.distributed._tensor._op_schema import OpStrategy, PlacementStrategy
from .basic_strategy import gen_einsum_strategies
from torch.distributed._tensor.ops.utils import (
    infer_broadcast_dims_map,
    map_placements_after_broadcast,
    register_op_strategy,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)
from torch.distributed.device_mesh import DeviceMesh

from ..utils import is_tensor_shardable

from chop.ir.graph import MaseMetadata

aten = torch.ops.aten


def transpose_strategy(meta: MaseMetadata, mesh: tuple) -> OpStrategy:

    parent_node = meta.node.args[0]
    self_strategy = parent_node.meta["mase"]["software"]["autosharding"]["op_strategy"]

    assert isinstance(self_strategy, OpStrategy)

    transpose_strategies = []
    for input_strategy in self_strategy.strategies:
        input_spec = input_strategy.output_spec
        # follow the input spec but transpose the Shard placements
        output_placements = [
            Shard(1 - p.dim) if isinstance(p, Shard) else p
            for p in input_spec.placements
        ]
        transpose_strategy = PlacementStrategy(
            output_specs=DTensorSpec(
                mesh=input_strategy.output_spec.mesh,
                placements=tuple(output_placements),
                tensor_meta=TensorMeta(
                    shape=meta["common"]["results"]["data_out_0"]["shape"],
                    stride=None,
                    dtype=meta["common"]["results"]["data_out_0"]["torch_dtype"],
                ),
            ),
            input_specs=(input_strategy.output_spec,),
        )
        transpose_strategies.append(transpose_strategy)

    return OpStrategy(strategies=transpose_strategies)


def _mm_like_strategy(mm_equation: str, meta: MaseMetadata, mesh: tuple) -> OpStrategy:
    self_shape, mat2_shape = [arg["shape"] for arg in meta["common"]["args"].values()]
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        assert strtg.input_specs is not None
        self_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]

        self_spec.tensor_meta = TensorMeta(
            shape=self_shape,
            stride=None,
            dtype=meta["common"]["args"]["data_in_0"]["torch_dtype"],
        )
        mat2_spec.tensor_meta = TensorMeta(
            shape=mat2_shape,
            stride=None,
            dtype=meta["common"]["args"]["data_in_1"]["torch_dtype"],
        )
        strtg.output_spec.tensor_meta = TensorMeta(
            shape=meta["common"]["results"]["data_out_0"]["shape"],
            stride=None,
            dtype=meta["common"]["results"]["data_out_0"]["torch_dtype"],
        )

        if is_tensor_shardable(self_shape, self_spec) and is_tensor_shardable(
            mat2_shape, mat2_spec
        ):
            filtered_strategies.append(strtg)

    mm_strategy.strategies = filtered_strategies

    return mm_strategy


def _addmm_like_strategy(
    mm_equation: str, meta: MaseMetadata, mesh: tuple
) -> OpStrategy:

    self_shape, mat1_shape, mat2_shape = [
        arg["shape"] for arg in meta["common"]["args"].values()
    ]

    mm_out_shape = torch.Size(
        [
            mat2_shape[-1] if i == len(mat1_shape) - 1 else dim_size
            for i, dim_size in enumerate(mat1_shape)
        ]
    )
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        # construct new strategy by consider the self arg
        assert strtg.input_specs is not None
        mat1_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        out_spec = strtg.output_spec

        # self arg's spec should follow the output of mm, but need
        # to consider broadcast for the self arg
        broadcast_dims_map = infer_broadcast_dims_map(mm_out_shape, self_shape)
        self_placements = map_placements_after_broadcast(
            out_spec.placements, mm_out_shape, broadcast_dims_map
        )
        self_spec = DTensorSpec(mesh=mesh, placements=self_placements)

        self_spec.tensor_meta = TensorMeta(
            shape=self_shape,
            stride=None,
            dtype=meta["common"]["args"]["data_in_0"]["torch_dtype"],
        )
        mat1_spec.tensor_meta = TensorMeta(
            shape=mat1_shape,
            stride=None,
            dtype=meta["common"]["args"]["data_in_1"]["torch_dtype"],
        )
        mat2_spec.tensor_meta = TensorMeta(
            shape=mat2_shape,
            stride=None,
            dtype=meta["common"]["args"]["data_in_2"]["torch_dtype"],
        )
        strtg.output_spec.tensor_meta = TensorMeta(
            shape=meta["common"]["results"]["data_out_0"]["shape"],
            stride=None,
            dtype=meta["common"]["results"]["data_out_0"]["torch_dtype"],
        )

        if is_tensor_shardable(mat1_shape, mat1_spec) and is_tensor_shardable(
            mat2_shape, mat2_spec
        ):
            # update input specs with new self spec
            strtg.input_specs = (self_spec, mat1_spec, mat2_spec)
            filtered_strategies.append(strtg)

    mm_strategy.strategies = filtered_strategies

    return mm_strategy


def mm_strategy(meta: MaseMetadata, mesh: tuple) -> OpStrategy:
    return _mm_like_strategy("mk,kn->mn", meta, mesh)


def addmm_strategy(meta: MaseMetadata, mesh: tuple) -> OpStrategy:
    return _addmm_like_strategy("mk,kn->mn", meta, mesh)


def bmm_strategy(meta: MaseMetadata, mesh: tuple) -> OpStrategy:
    return _mm_like_strategy("bmk,bkn->bmn", meta, mesh)


def baddmm_strategy(meta: MaseMetadata, mesh: tuple) -> OpStrategy:
    return _addmm_like_strategy("bmk,bkn->bmn", meta, mesh)
