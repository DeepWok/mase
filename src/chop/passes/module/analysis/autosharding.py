import torch
import torch.nn.functional as F

import numpy as np
import cvxpy as cp
from copy import copy
from collections import OrderedDict

import vllm
from vllm.attention import Attention as VllmAttention

from chop.tools import get_logger
from chop.distributed.utils import rlog

from .cost_modelling import (
    _get_compute_cost_from_layer,
    _get_intra_op_comms_cost,
    _get_resharding_cost_matrix,
    _get_memory_cost_from_layer,
)

VllmLinear = vllm.model_executor.layers.linear.LinearBase

logger = get_logger(__name__)
logger.setLevel("WARNING")

STRATEGY_MAP = OrderedDict(
    {
        VllmLinear: (
            "replicated",
            "column",
            "row",
            "data",
        ),
        VllmAttention: (
            "replicated",
            "head",
        ),
        type(None): None,
    }
)


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

    return tuple(size)


def _linearize_resharding_cost(
    opt_var,
    parent_opt_var,
    resharding_costs,
):
    # Flatten resharding matrix
    resharding_costs = resharding_costs.flatten()

    # Formulate linearized variable for resharding cost
    e_var = cp.Variable(resharding_costs.shape[0], boolean=True)
    expr = e_var.T @ resharding_costs
    constr = [
        cp.sum(e_var) == 1,
    ]

    # Constraints s.t. e_var = outer(opt_var, in_opt_var)
    indices = np.arange(e_var.shape[0])
    opt_indices, in_opt_indices = np.divmod(indices, parent_opt_var.shape[0])
    constr += [
        e_var <= opt_var[opt_indices],
        e_var <= parent_opt_var[in_opt_indices],
        e_var >= opt_var[opt_indices] + parent_opt_var[in_opt_indices] - 1,
    ]

    return expr, constr


def _get_memory_constraint(
    memory_constraint_terms: list,
    self_rank: int,
    pass_args: dict,
):
    budget = pass_args.get("gpu_memory_budget", None)

    if budget is None:
        raise ValueError("gpu_memory_budget is required for autosharding analysis")

    memory_available = torch.cuda.get_device_properties(self_rank).total_memory * budget

    mem_constr_expr = 0
    for i, (opt_var, mem_cost) in enumerate(memory_constraint_terms):
        mem_constr_expr += mem_cost @ opt_var

    return [mem_constr_expr <= memory_available]


def _formulate_ilp(
    model: torch.nn.Module,
    pass_args: dict,
):

    self_rank = torch.distributed.get_rank()
    data_size = pass_args.get("data_size", None)

    if data_size is None:
        raise ValueError("data_size is required for autosharding analysis")

    module_list = []
    module_strategies = []

    # ILP variables
    constr = []
    expr = 0
    megatron_soln = 0
    megatron_mem_cost = 0

    bad_soln = 0
    bad_soln_memory_cost = 0

    # List of tuples: (opt_var, memory_cost)
    memory_constr_terms = []

    for name, layer in model.named_modules():

        # Skip non-leaf modules
        if len(list(layer.children())) > 0:
            continue
        rlog(logger, self_rank, f"Parsing layer {layer.__class__.__name__}")

        # Check if matches with one of the supported layer types
        for layer_type, layer_strategies in STRATEGY_MAP.items():
            if isinstance(layer, layer_type):
                break

        if layer_type is None or layer_strategies is None:
            continue

        # Register layer and instantiate optimization variable
        # ============================
        module_list.append(layer)
        module_strategies.append(layer_strategies)

        opt_var = cp.Variable(len(layer_strategies), boolean=True)
        setattr(layer, "opt_var", opt_var)
        constr += [
            cp.sum(opt_var) == 1,
        ]

        # Calculate Megatron solution for comparison
        megatron_opt_var = np.zeros(len(layer_strategies))
        bad_soln_opt_var = np.zeros(len(layer_strategies))

        if "attn.c_attn" in name:
            megatron_opt_var[1] = 1  # column
            bad_soln_opt_var[1] = 1  # column
        elif "attn.attn" in name:
            megatron_opt_var[1] = 1  # head
            bad_soln_opt_var[1] = 1  # head
        elif "attn.c_proj" in name:
            megatron_opt_var[2] = 1  # row
            bad_soln_opt_var[1] = 1  # column
        elif "mlp.c_fc" in name:
            megatron_opt_var[1] = 1  # column
            bad_soln_opt_var[1] = 1  # column
        elif "mlp.c_proj" in name:
            megatron_opt_var[2] = 1  # row
            bad_soln_opt_var[2] = 1  # column
        else:
            raise ValueError(f"Unsupported layer name: {name}")

        setattr(layer, "megatron_opt_var", megatron_opt_var)
        setattr(layer, "bad_soln_opt_var", bad_soln_opt_var)

        # Consider compute cost
        # ============================
        compute_cost = _get_compute_cost_from_layer(
            layer,
            layer_strategies,
            data_size=data_size,
            benchmarking_device=self_rank,
        )

        # Consider intra operator comms cost
        # ============================

        comms_cost = _get_intra_op_comms_cost(
            layer_strategies=tuple(layer_strategies),
            output_shape=_get_output_shape_from_layer_type(layer, data_size),
            benchmarking_device=self_rank,
        )

        expr += (compute_cost + comms_cost) @ opt_var
        megatron_soln += (compute_cost + comms_cost) @ megatron_opt_var
        bad_soln += (compute_cost + comms_cost) @ bad_soln_opt_var

        # Consider memory cost
        # ============================

        mem_cost = _get_memory_cost_from_layer(
            layer,
            layer_strategies,
            benchmarking_device=self_rank,
        )

        memory_constr_terms.append((opt_var, mem_cost))
        megatron_mem_cost += mem_cost @ megatron_opt_var

        bad_soln_memory_cost += mem_cost @ bad_soln_opt_var

        # Consider resharding cost
        # ============================

        # Skip if no parent module
        if len(module_list) <= 1:
            continue

        parent_module = module_list[-2]
        parent_strategies = module_strategies[-2]
        logger.info(
            f"Consider resharding cost between {parent_module.__class__.__name__} and {layer.__class__.__name__}"
        )

        parent_out_shape = _get_output_shape_from_layer_type(
            parent_module,
            data_size,
        )

        resharding_costs = _get_resharding_cost_matrix(
            layer_strategies=layer_strategies,
            parent_strategies=parent_strategies,
            parent_out_shape=parent_out_shape,
            benchmarking_device=self_rank,
        )

        resharding_term, resharding_constraints = _linearize_resharding_cost(
            opt_var,
            parent_module.opt_var,
            resharding_costs,
        )
        expr += resharding_term
        constr += resharding_constraints

        # Add Megatron solution for comparison
        megatron_resharding_term = (
            parent_module.megatron_opt_var @ resharding_costs @ megatron_opt_var
        )
        megatron_soln += megatron_resharding_term

        bad_soln_resharding_term = (
            parent_module.bad_soln_opt_var @ resharding_costs @ bad_soln_opt_var
        )
        bad_soln += bad_soln_resharding_term

    # After processing all layers, consider memory constraints
    # ============================

    mem_constr = _get_memory_constraint(
        memory_constraint_terms=memory_constr_terms,
        self_rank=self_rank,
        pass_args=pass_args,
    )
    constr += mem_constr

    return (
        cp.Problem(cp.Minimize(expr), constr),
        (megatron_soln, megatron_mem_cost),
        mem_constr,
    )


def _get_sharding_config(model):
    self_rank = torch.distributed.get_rank()

    sharding_config = {}
    for layer in model.modules():

        # Skip non-leaf modules
        if len(list(layer.children())) > 0:
            continue

        # Check if matches with one of the supported layer types
        for layer_type, layer_strategies in STRATEGY_MAP.items():
            if isinstance(layer, layer_type):
                break

        if layer_type is None or layer_strategies is None:
            continue

        opt_var_value = layer.opt_var.value
        strategy_idx = np.where(opt_var_value)[0][0]
        strategy = layer_strategies[strategy_idx]

        sharding_config[layer.prefix] = strategy

    return sharding_config


def autosharding_module_analysis_pass(model, pass_args={}):
    problem, megatron, mem_constr = _formulate_ilp(model, pass_args)
    megatron_soln, megatron_mem_cost = megatron
    problem.solve(
        verbose=pass_args.get(f"debug", False),
        scipy_options={
            "disp": pass_args.get(f"debug", False),
            "time_limit": pass_args.get("time_limit", None),
            "mip_rel_gap": pass_args.get("mip_rel_gap", 0) / 100,
        },
    )

    sharding_config = _get_sharding_config(model)

    memory_available = torch.cuda.get_device_properties(
        torch.distributed.get_rank()
    ).total_memory * pass_args.get("gpu_memory_budget")

    return model, {
        "sharding_config": sharding_config,
    }
