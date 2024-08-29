import math
import numpy as np
from copy import copy
from functools import lru_cache

import torch
from torch.nn import functional as F

import vllm
from vllm.attention import Attention as VllmAttention

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

    return out, np.mean(elapsed[warmup_iters:])


def allreduce_cost(
    bytes_gb: float,
    intra_device_latency: float,
    intra_device_bandwidth: float,
) -> float:
    world_size = torch.distributed.get_world_size()
    mesh_dim_bandwidth = intra_device_bandwidth
    # allreduce have almost 2x comm bytes compare to allgather/reduce_scatter
    num_hops = 2 * world_size - 1

    latency = 6.6 + num_hops * intra_device_latency
    bw = (bytes_gb * num_hops / world_size) / mesh_dim_bandwidth
    return latency + bw * 1e6


def allgather_cost(
    bytes_gb: float,
    intra_device_latency: float,
    intra_device_bandwidth: float,
) -> float:
    world_size = torch.distributed.get_world_size()
    num_hops = world_size - 1
    latency = 6.6 + num_hops * intra_device_latency
    bw = (bytes_gb * num_hops / world_size) / intra_device_bandwidth
    return latency + bw * 1e6


def _get_output_shape_from_layer_type(
    layer: torch.nn.Module,
    data_size: int,
):
    if isinstance(layer, vllm.model_executor.layers.linear.LinearBase):
        return torch.Size([data_size, layer.weight.shape[0]])
    if isinstance(layer, VllmAttention):
        return torch.Size([data_size, layer.impl.head_size * layer.impl.num_heads])
    else:
        raise ValueError(f"Unsupported layer type: {layer.__class__.__name__}")


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
        local_shape = copy(global_shape)

        if strategy == "replicated":
            pass
        elif strategy == "column":
            local_shape = torch.Size([global_shape[0], global_shape[1] // world_size])

        elapsed = _cached_attention_cost_from_local_shapes(
            local_shape,
            repeat=repeat,
            warmup_iters=warmup_iters,
        )

        cost_vector.append(elapsed)

    return np.array(cost_vector)


def _get_compute_cost_from_layer(layer, layer_strategies, data_size):
    if isinstance(layer, vllm.model_executor.layers.linear.LinearBase):
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
    bw = pass_args.get("intra_device_bandwidth", None)
    lat = pass_args.get("intra_device_latency", None)
    data_size = pass_args.get("data_size", None)

    if bw is None:
        raise ValueError("intra_device_bandwidth is not provided")
    if lat is None:
        raise ValueError("intra_device_latency is not provided")
    if data_size is None:
        raise ValueError("data_size is not provided")

    comms_cost = np.zeros(len(layer_strategies))
    for idx, strategy in enumerate(layer_strategies):
        if strategy == "row":
            comms_cost[idx] = allreduce_cost(
                bytes_gb=math.prod(_get_output_shape_from_layer_type(layer, data_size))
                * 4
                / 1e9,
                intra_device_latency=lat,
                intra_device_bandwidth=bw,
            )

    return comms_cost * 1e-6  # convert back to seconds


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
    bw = pass_args.get("intra_device_bandwidth", None)
    lat = pass_args.get("intra_device_latency", None)
    data_size = pass_args.get("data_size", None)

    if bw is None:
        raise ValueError("intra_device_bandwidth is not provided")
    if lat is None:
        raise ValueError("intra_device_latency is not provided")
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
        cost = allgather_cost(
            bytes_gb=world_size
            * math.prod(_get_output_shape_from_layer_type(parent_module, data_size))
            * 4
            / 1e9,
            intra_device_latency=lat,
            intra_device_bandwidth=bw,
        )
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

    return resharding_costs * 1e-6  # convert back to seconds
