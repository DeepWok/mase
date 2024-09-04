import os
import math
import gc
import numpy as np
from copy import copy
from functools import lru_cache

import torch
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, set_start_method


import vllm
from vllm.attention import Attention as VllmAttention
from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    ColumnParallelLinear,
    RowParallelLinear,
    DataParallelLinear,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)

VllmLinear = vllm.model_executor.layers.linear.LinearBase


# Utilities
# ================================


def _linear_cls_from_config(config: str):
    if config == "replicated":
        return ReplicatedLinear
    if config == "column":
        return ColumnParallelLinear
    if config == "row":
        return RowParallelLinear
    if config == "data":
        return DataParallelLinear

    raise ValueError(f"Unknown linear config: {config}")


def _profile_op(
    op: str,
    fn: callable,
    shape: tuple,
    repeat: int,
    warmup_iters: int,
    benchmarking_device: int = 0,
    extra_args: list = [],
):
    """
    Profile op ``repeat`` times with ``warmup_iters`` warmup iterations.
    Generate random input tensors of shape ``shape`` and pass them to the function ``fn`` in each iteration.

    Args:
        op (str): _description_
        fn (callable): _description_
        shape (tuple): _description_
        repeat (int): _description_
        warmup_iters (int): _description_
        benchmarking_device (int, optional): _description_. Defaults to 0.
        extra_args (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    start_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(repeat)
    ]
    end_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(repeat)
    ]

    for idx in range(repeat):
        if op == "linear":
            input_ = torch.randn(shape).to(f"cuda:{benchmarking_device}")
            args = [input_] + extra_args
        elif op == "attention":
            local_query = torch.randn(shape).to(f"cuda:{benchmarking_device}")
            local_key = torch.randn(shape).to(f"cuda:{benchmarking_device}")
            local_value = torch.randn(shape).to(f"cuda:{benchmarking_device}")
            args = [
                local_query,
                local_key,
                local_value,
                None,  # benchmark without KV cache
            ] + extra_args
        elif op == "allreduce":
            local_tensor = torch.randn(shape).to(f"cuda:{benchmarking_device}")
            args = [local_tensor]
        elif op == "allgather":
            local_tensor = torch.randn(shape).to(f"cuda:{benchmarking_device}")
            args = [local_tensor, -1]
        else:
            raise ValueError(f"Unknown op: {op}")

        start_event[idx].record()
        out = fn(*args)
        end_event[idx].record()
    torch.cuda.synchronize(device=f"cuda:{benchmarking_device}")

    elapsed = [start_event[idx].elapsed_time(end_event[idx]) for idx in range(repeat)]

    return out, np.mean(elapsed[warmup_iters:]), elapsed


@lru_cache(maxsize=128, typed=False)
def allreduce_cost(
    output_shape: tuple,
    repeat: int = 100,
    warmup_iters: int = 5,
    benchmarking_device: int = 0,
) -> float:
    _, cost, elapsed_times = _profile_op(
        op="allreduce",
        fn=tensor_model_parallel_all_reduce,
        shape=output_shape,
        repeat=repeat,
        warmup_iters=warmup_iters,
        benchmarking_device=benchmarking_device,
    )

    return cost


@lru_cache(maxsize=128, typed=False)
def allgather_cost(
    local_shape: tuple,
    repeat: int = 100,
    warmup_iters: int = 5,
    benchmarking_device: int = 0,
) -> float:
    _, cost, elapsed_times = _profile_op(
        op="allgather",
        fn=tensor_model_parallel_all_gather,
        shape=local_shape,
        repeat=repeat,
        warmup_iters=warmup_iters,
        benchmarking_device=benchmarking_device,
    )

    return cost


# Compute cost
# ================================


@lru_cache(maxsize=128, typed=False)
def _cached_linear_cost_from_local_shapes(
    type: str,
    data_size: int,
    input_size: int,
    output_size: int,
    repeat: int = 100,
    warmup_iters: int = 2,
    benchmarking_device: int = 0,
):
    cls = _linear_cls_from_config(type)

    layer = cls(
        input_size=input_size,
        output_size=output_size,
    )

    local_shape = (data_size, input_size)
    if type == "data":
        local_shape = (data_size // torch.distributed.get_world_size(), input_size)
    elif type == "row":
        local_shape = (data_size, input_size // torch.distributed.get_world_size())
    elif type in ["replicated", "column"]:
        pass
    else:
        raise ValueError(f"Unknown type: {type}")

    _, elapsed, elapsed_list = _profile_op(
        op="linear",
        fn=layer,
        shape=local_shape,
        repeat=repeat,
        warmup_iters=warmup_iters,
        benchmarking_device=benchmarking_device,
    )

    return elapsed, elapsed_list


def _get_linear_compute_cost(
    layer: torch.nn.Module,
    layer_strategies: tuple,
    data_size: int,
    repeat: int = 100,
    warmup_iters: int = 2,
    benchmarking_device: int = 0,
):

    input_size = layer.input_size
    output_size = layer.output_size

    cost_vector = []
    for strategy in layer_strategies:

        # Create local tensors
        elapsed, elapsed_list = _cached_linear_cost_from_local_shapes(
            type=strategy,
            data_size=data_size,
            input_size=input_size,
            output_size=output_size,
            repeat=repeat,
            warmup_iters=warmup_iters,
            benchmarking_device=benchmarking_device,
        )

        cost_vector.append(elapsed)

    return np.array(cost_vector)


@lru_cache(maxsize=128, typed=False)
def _cached_attention_cost_from_local_shapes(
    data_size: int,
    num_heads: int,
    head_size: int,
    repeat: int = 100,
    warmup_iters: int = 2,
    benchmarking_device: int = 0,
):
    local_shape = torch.Size([data_size, head_size * num_heads])

    attn_meta = AttentionMetadata(
        num_prefills=9,
        num_prefill_tokens=data_size,
        num_decode_tokens=0,
        slot_mapping=None,
    )
    attn_layer = VllmAttention(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0,
    )

    _, elapsed, _ = _profile_op(
        op="attention",
        fn=attn_layer,
        shape=local_shape,
        repeat=repeat,
        warmup_iters=warmup_iters,
        benchmarking_device=benchmarking_device,
        extra_args=[attn_meta],
    )

    return elapsed


def _get_attention_compute_cost(
    layer: torch.nn.Module,
    layer_strategies: tuple,
    data_size: int,
    repeat: int = 100,
    warmup_iters: int = 2,
    benchmarking_device: int = 0,
):

    cost_vector = []
    for strategy in layer_strategies:

        if strategy == "replicated":
            num_heads = layer.impl.num_heads
        elif strategy == "head":
            num_heads = layer.impl.num_heads // torch.distributed.get_world_size()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        elapsed = _cached_attention_cost_from_local_shapes(
            data_size=data_size,
            num_heads=num_heads,
            head_size=layer.impl.head_size,
            repeat=repeat,
            warmup_iters=warmup_iters,
            benchmarking_device=benchmarking_device,
        )

        cost_vector.append(elapsed)

    return np.array(cost_vector)


def _get_compute_cost_from_layer(
    layer,
    layer_strategies,
    data_size,
    benchmarking_device: int = 0,
):
    if isinstance(layer, VllmLinear):
        return _get_linear_compute_cost(
            layer,
            layer_strategies,
            data_size,
            benchmarking_device=benchmarking_device,
        )
    if isinstance(layer, VllmAttention):
        return _get_attention_compute_cost(
            layer,
            layer_strategies,
            data_size,
            benchmarking_device=benchmarking_device,
        )
    else:
        raise ValueError(f"Unsupported layer type: {layer.__class__.__name__}")


@lru_cache(maxsize=128, typed=False)
def _get_intra_op_comms_cost(
    layer_strategies: tuple,
    output_shape: tuple,
    benchmarking_device: int = 0,
):
    comms_cost = np.zeros(len(layer_strategies))
    for idx, strategy in enumerate(layer_strategies):
        if strategy == "row":
            comms_cost[idx] = allreduce_cost(
                output_shape=output_shape,
                benchmarking_device=benchmarking_device,
            )

    return comms_cost


# Resharding cost
# ================================


def _get_resharding_cost(
    module_strategy: str,
    parent_out_shape: tuple,
    parent_strategy: str,
    benchmarking_device: int = 0,
) -> float:

    # Strategies which always return RR sharding
    if parent_strategy in ["replicated", "row"]:
        return 0

    world_size = torch.distributed.get_world_size()

    # all gather operation
    skip_allgather = (
        (
            # Column parallel linear -> Row parallel linear (Megatron-LM)
            parent_strategy == "column"
            and module_strategy == "row"
        )
        or (
            # Column parallel linear -> Head parallel attention (Megatron-LM)
            parent_strategy == "column"
            and module_strategy == "head"
        )
        or (
            # Head parallel attention -> Row parallel linear (Megatron-LM)
            parent_strategy == "head"
            and module_strategy == "row"
        )
        or (
            # Data parallel linear -> Data parallel linear
            parent_strategy == "data"
            and module_strategy == "data"
        )
    )

    if not skip_allgather:
        local_shape = [parent_out_shape[0], parent_out_shape[1] // world_size]
        cost = allgather_cost(
            local_shape=tuple(local_shape),
            benchmarking_device=benchmarking_device,
        )
    else:
        cost = 0

    return cost


@lru_cache(maxsize=128, typed=False)
def _get_resharding_cost_matrix(
    layer_strategies,
    parent_strategies,
    parent_out_shape,
    benchmarking_device: int = 0,
):

    resharding_costs = np.zeros([len(parent_strategies), len(layer_strategies)])
    for module_strategy_idx, module_strategy in enumerate(layer_strategies):
        for parent_strategy_idx, parent_strategy in enumerate(parent_strategies):
            resharding_costs[parent_strategy_idx, module_strategy_idx] = (
                _get_resharding_cost(
                    module_strategy,
                    parent_out_shape,
                    parent_strategy,
                    benchmarking_device=benchmarking_device,
                )
            )

    return resharding_costs


# Memory cost
# ================================


def _get_gpu_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@lru_cache(maxsize=128, typed=False)
def _cached_get_linear_memory_cost(
    input_size,
    output_size,
    bias,
    strategies,
    benchmarking_device: int = 0,
):
    cost_vector = []
    peak_mems = []
    for strategy in strategies:
        cls = _linear_cls_from_config(strategy)

        # Clear cache and reset stats
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        # Instantiate layer to measure memory usage
        start_memory = _get_gpu_memory_usage()
        _ = cls(
            input_size=input_size,
            output_size=output_size,
            bias=bias is not None,
        ).to(f"cuda:{benchmarking_device}")
        end_memory = _get_gpu_memory_usage()

        # Record cost
        cost_vector.append(end_memory - start_memory)
        peak_mems.append(torch.cuda.max_memory_allocated())

    return cost_vector


def _get_memory_cost_from_layer(
    layer,
    layer_strategies,
    benchmarking_device: int = 0,
):
    if isinstance(layer, VllmLinear):
        return _cached_get_linear_memory_cost(
            input_size=layer.input_size,
            output_size=layer.output_size,
            bias=layer.bias is not None,
            strategies=tuple(layer_strategies),
            benchmarking_device=benchmarking_device,
        )
    elif isinstance(layer, VllmAttention):
        return np.zeros(len(layer_strategies))
    else:
        raise ValueError(f"Unsupported layer type: {layer.__class__.__name__}")
