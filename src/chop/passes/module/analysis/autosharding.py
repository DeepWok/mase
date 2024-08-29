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
    _get_resharding_cost_matrix,
    _get_intra_op_comms_cost,
)

VllmLinear = vllm.model_executor.layers.linear.LinearBase

logger = get_logger(__name__)
logger.setLevel("DEBUG")

STRATEGY_MAP = OrderedDict(
    {
        VllmLinear: [
            "replicated",
            "column",
            "row",
            "data",
        ],
        VllmAttention: [
            "replicated",
            "head",
        ],
        type(None): None,
    }
)


def _linearize_resharding_cost(
    opt_var,
    parent_opt_var,
    resharding_costs,
    pass_args,
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

    for layer in model.modules():

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

        # Consider compute cost
        # ============================
        compute_cost = _get_compute_cost_from_layer(
            layer,
            layer_strategies,
            data_size=data_size,
        )
        expr += compute_cost @ opt_var

        # Consider intra operator comms cost
        # ============================
        comms_cost = _get_intra_op_comms_cost(
            layer,
            layer_strategies,
            pass_args=pass_args,
        )
        expr += comms_cost @ opt_var

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

        resharding_costs = _get_resharding_cost_matrix(
            layer, layer_strategies, parent_module, parent_strategies, pass_args
        )

        resharding_term, resharding_constraints = _linearize_resharding_cost(
            opt_var, parent_module.opt_var, resharding_costs, pass_args
        )
        expr += resharding_term
        constr += resharding_constraints

    return cp.Problem(cp.Minimize(expr), constr)


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
    problem = _formulate_ilp(model, pass_args)
    problem.solve(
        verbose=True,
        scipy_options={
            "disp": pass_args.get(f"debug", False),
            "time_limit": pass_args.get("time_limit", None),
            "mip_rel_gap": pass_args.get("mip_rel_gap", 0) / 100,
        },
    )

    sharding_config = _get_sharding_config(model)

    return model, {
        "sharding_config": sharding_config,
    }
