# Adapted from Pytorch Distributed DTensor API.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/ops/matrix_ops.py

import torch
from torch.distributed._tensor._op_schema import (
    OpStrategy,
    PlacementStrategy,
    PlacementList,
)
from torch.distributed._tensor.placement_types import Replicate, Shard, Placement
from .basic_strategy import gen_einsum_strategies
from torch.distributed._tensor.ops.utils import (
    infer_broadcast_dims_map,
    map_placements_after_broadcast,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Shard,
    TensorMeta,
)

from ..utils import is_tensor_shardable

from chop.ir.graph import MaseMetadata


def transpose_strategy(
    meta: MaseMetadata,
    mesh: tuple,
) -> OpStrategy:

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


def _mm_like_strategy(
    mm_equation: str,
    meta: MaseMetadata,
    mesh: tuple,
) -> OpStrategy:
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
    mm_equation: str,
    meta: MaseMetadata,
    mesh: tuple,
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


def mm_strategy(
    meta: MaseMetadata,
    mesh: tuple,
) -> OpStrategy:
    return _mm_like_strategy("mk,kn->mn", meta, mesh)


def addmm_strategy(
    meta: MaseMetadata,
    mesh: tuple,
) -> OpStrategy:
    return _addmm_like_strategy("mk,kn->mn", meta, mesh)


def bmm_strategy(
    meta: MaseMetadata,
    mesh: tuple,
) -> OpStrategy:
    return _mm_like_strategy("bmk,bkn->bmn", meta, mesh)


def baddmm_strategy(
    meta: MaseMetadata,
    mesh: tuple,
) -> OpStrategy:
    return _addmm_like_strategy("bmk,bkn->bmn", meta, mesh)


def scaled_dot_product_flash_attention_strategy(
    meta: MaseMetadata,
    mesh: tuple,
) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    # TODO: sdpa might be a good candidate for us to explore decomposed sharding propagation
    # as it involves: matmul, pointwise, reduction ops together.
    return_debug_mask = len(op_schema.args_schema) >= 6 and op_schema.args_schema[5]
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape
    qkv_shape = q_input_strategy.shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the spda case, we have 3 valid tensor outputs and 3 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
        Replicate(),
        Replicate(),
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        Replicate(),
        Replicate(),
        Replicate(),
        Replicate(),
    ]
    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the num of head dim
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim
    logsumexp_sharding = Shard(1)  # num head dim
    if return_debug_mask:
        debug_attn_mask_sharding: Placement = Shard(1)  # num head dim
    else:
        # empty debug mask, replicated
        debug_attn_mask_sharding = Replicate()

    num_heads_dim_sharding: PlacementList = [
        output_sharding,
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_sharding,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Context Parallelism: shards on the sequence dim
    single_mesh_dim_strategies.append(
        [
            Shard(2),  # output
            Shard(2),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            None,  # philox_seed
            None,  # philox_offset
            Shard(2),  # debugattn
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
        ]
    )
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )
