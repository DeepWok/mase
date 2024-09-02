import os
import math
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

VllmLinear = vllm.model_executor.layers.linear.LinearBase


# Utilities
# ================================


def _profile_op(
    fn: callable,
    args: list,
    repeat: int,
    warmup_iters: int,
):
    start_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(repeat)
    ]
    end_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(repeat)
    ]

    for idx in range(repeat):
        start_event[idx].record()
        out = fn(*args)
        end_event[idx].record()
    torch.cuda.synchronize(device=f"cuda:0")

    elapsed = [start_event[idx].elapsed_time(end_event[idx]) for idx in range(repeat)]

    return out, np.mean(elapsed[warmup_iters:]) * 1e-3  # convert back to seconds


def _profile_distributed_op(
    rank,
    world_size,
    result_queue,
    repeat,
    warmup_iters,
    op,
    global_shape,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    os.environ["RANK"] = str(rank)

    # Initialize
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    torch.cuda.set_device(device)

    start_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(repeat)
    ]
    end_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(repeat)
    ]

    for idx in range(repeat):
        if op == "allgather":
            output_tensor = torch.zeros(global_shape, device=device)
            local_shape = [global_shape[0], global_shape[1] // world_size]
            local_tensor = torch.randn(local_shape, device=device)

            dist.barrier()
            start_event[idx].record()

            dist.all_gather_into_tensor(output_tensor, local_tensor)
            output_tensor = output_tensor.movedim(0, 1)
            output_tensor = output_tensor.reshape(global_shape)

        elif op == "allreduce":
            local_tensor = torch.randn(global_shape, device=device)

            dist.barrier()
            start_event[idx].record()

            dist.all_reduce(local_tensor)

        dist.barrier()
        end_event[idx].record()

        torch.cuda.synchronize(device=device)

    elapsed = [start_event[idx].elapsed_time(end_event[idx]) for idx in range(repeat)]

    avg = sum(elapsed[warmup_iters:]) / len(elapsed[warmup_iters:])

    if rank == 0:
        result_queue.put(avg)

    dist.barrier()
    dist.destroy_process_group()


def allreduce_cost(
    output_shape: list,
    repeat: int = 100,
    warmup_iters: int = 5,
) -> float:
    ds, hs = output_shape

    intercept = 0.40594790939481484

    coeff = [
        0.0,
        -0.00019876370905316763,
        -4.174260473864464e-06,
        4.019442387061491e-08,
        6.210839534401708e-07,
        4.909228531291631e-11,
    ]

    cost = (
        intercept
        + coeff[0]
        + (coeff[1] * ds)
        + (coeff[2] * hs)
        + (coeff[3] * ds**2)
        + (coeff[4] * ds * hs)
        + (coeff[5] * hs**2)
    )

    return cost * 1e-3


def allgather_cost(
    output_shape: list,
    repeat: int = 100,
    warmup_iters: int = 5,
) -> float:
    ds, hs = output_shape

    intercept = 0.478361915750253

    coeff = [
        0,
        -0.00025625419990716485,
        -1.9612017748514218e-05,
        4.892589021040619e-08,
        3.375990357833703e-07,
        5.192329766543819e-10,
    ]

    cost = (
        intercept
        + coeff[0]
        + (coeff[1] * ds)
        + (coeff[2] * hs)
        + (coeff[3] * ds**2)
        + (coeff[4] * ds * hs)
        + (coeff[5] * hs**2)
    )

    return cost * 1e-3  # convert back to seconds


def _get_output_shape_from_layer_type(
    layer: torch.nn.Module,
    data_size: int,
):
    if isinstance(layer, VllmLinear):
        size = torch.Size([data_size, layer.weight.shape[0]])
    elif isinstance(layer, VllmAttention):
        size = torch.Size([data_size, layer.impl.head_size * layer.impl.num_heads])
    else:
        raise ValueError(f"Unsupported layer type: {layer.__class__.__name__}")

    return list(size)


# Compute cost
# ================================


@lru_cache(maxsize=128, typed=False)
def _cached_linear_cost_from_local_shapes(
    local_weight_shape: torch.Size,
    local_bias_shape: torch.Size,
    local_input_shape: torch.Size,
    repeat: int = 100,
    warmup_iters: int = 2,
):
    local_weights = torch.randn(local_weight_shape).to("cuda:0")
    local_bias = torch.randn(local_bias_shape).to("cuda:0")
    local_input = torch.randn(local_input_shape).to("cuda:0")

    _, elapsed = _profile_op(
        fn=F.linear,
        args=[local_input, local_weights, local_bias],
        repeat=repeat,
        warmup_iters=warmup_iters,
    )

    return elapsed


def _get_linear_compute_cost(
    layer: torch.nn.Module,
    layer_strategies: list,
    data_size: int,
    repeat: int = 100,
    warmup_iters: int = 2,
):

    world_size = torch.distributed.get_world_size()

    # Global shapes
    global_weight_shape = layer.weight.shape
    global_bias_shape = layer.bias.shape
    global_input_shape = torch.Size([data_size, global_weight_shape[1]])

    cost_vector = []
    for strategy in layer_strategies:

        # Default values for local tensors
        # (taken for replicated strategy)
        local_weight_shape = copy(global_weight_shape)
        local_bias_shape = copy(global_bias_shape)
        local_input_shape = copy(global_input_shape)

        if strategy == "replicated":
            pass
        elif strategy == "column":
            local_weight_shape = torch.Size(
                [global_weight_shape[0] // world_size, global_weight_shape[1]]
            )
            local_bias_shape = torch.Size([global_bias_shape[0] // world_size])
        elif strategy == "row":
            local_input_shape = torch.Size(
                [global_input_shape[0], global_input_shape[1] // world_size]
            )
            local_weight_shape = torch.Size(
                [global_weight_shape[0], global_weight_shape[1] // world_size]
            )
        elif strategy == "data":
            local_input_shape = torch.Size(
                [global_input_shape[0] // world_size, global_input_shape[1]]
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Create local tensors
        elapsed = _cached_linear_cost_from_local_shapes(
            local_weight_shape,
            local_bias_shape,
            local_input_shape,
            repeat=repeat,
            warmup_iters=warmup_iters,
        )

        cost_vector.append(elapsed)

    return np.array(cost_vector)


@lru_cache(maxsize=128, typed=False)
def _cached_attention_cost_from_local_shapes(
    local_shape: torch.Size,
    repeat: int = 100,
    warmup_iters: int = 2,
):
    local_query = torch.randn(local_shape).to("cuda:0")
    local_key = torch.randn(local_shape).to("cuda:0")
    local_value = torch.randn(local_shape).to("cuda:0")

    _, elapsed = _profile_op(
        fn=F.scaled_dot_product_attention,
        args=[local_query, local_key, local_value],
        repeat=repeat,
        warmup_iters=warmup_iters,
    )

    return elapsed


def _get_attention_compute_cost(
    layer: torch.nn.Module,
    layer_strategies: list,
    data_size: int,
    repeat: int = 100,
    warmup_iters: int = 2,
):

    world_size = torch.distributed.get_world_size()

    global_shape = torch.Size([data_size, layer.impl.head_size * layer.impl.num_heads])

    cost_vector = []
    for strategy in layer_strategies:

        if strategy == "replicated":
            local_shape = copy(global_shape)
        elif strategy == "head":
            local_shape = torch.Size([global_shape[0], global_shape[1] // world_size])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        elapsed = _cached_attention_cost_from_local_shapes(
            local_shape,
            repeat=repeat,
            warmup_iters=warmup_iters,
        )

        cost_vector.append(elapsed)

    return np.array(cost_vector)


def _get_compute_cost_from_layer(layer, layer_strategies, data_size):
    if isinstance(layer, VllmLinear):
        return _get_linear_compute_cost(
            layer,
            layer_strategies,
            data_size,
        )
    if isinstance(layer, VllmAttention):
        return _get_attention_compute_cost(
            layer,
            layer_strategies,
            data_size,
        )
    else:
        raise ValueError(f"Unsupported layer type: {layer.__class__.__name__}")


def _get_intra_op_comms_cost(
    layer: torch.nn.Module,
    layer_strategies: list,
    pass_args: dict,
):
    data_size = pass_args.get("data_size", None)

    if data_size is None:
        raise ValueError("data_size is not provided")

    comms_cost = np.zeros(len(layer_strategies))
    for idx, strategy in enumerate(layer_strategies):
        if strategy == "row":
            out_shape = _get_output_shape_from_layer_type(layer, data_size)
            comms_cost[idx] = allreduce_cost(output_shape=out_shape)

    return comms_cost


# Resharding cost
# ================================


def _get_resharding_cost(
    layer: torch.nn.Module,
    module_strategy: str,
    parent_module: torch.nn.Module,
    parent_strategy: str,
    pass_args: dict,
) -> float:

    # Strategies which always return RR sharding
    if parent_strategy in ["replicated", "row"]:
        return 0

    world_size = torch.distributed.get_world_size()
    data_size = pass_args.get("data_size", None)

    if data_size is None:
        raise ValueError("data_size is not provided")

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
        out_shape = _get_output_shape_from_layer_type(parent_module, data_size)
        cost = allgather_cost(output_shape=out_shape)
    # elif...
    else:
        cost = 0

    return cost


def _get_resharding_cost_matrix(
    layer, layer_strategies, parent_module, parent_strategies, pass_args
):

    resharding_costs = np.zeros([len(parent_strategies), len(layer_strategies)])
    for module_strategy_idx, module_strategy in enumerate(layer_strategies):
        for parent_strategy_idx, parent_strategy in enumerate(parent_strategies):
            resharding_costs[parent_strategy_idx, module_strategy_idx] = (
                _get_resharding_cost(
                    layer,
                    module_strategy,
                    parent_module,
                    parent_strategy,
                    pass_args,
                )
            )

    return resharding_costs
