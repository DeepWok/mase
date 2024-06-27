import sys, pdb, traceback
import functools

import torch.nn as nn
import numpy as np
import cvxpy as cp

from chop.tools import get_logger

from .common import SpmdShard
from .alpa_layers import ALPA_FUNCTIONS, ALPA_METHODS
from .alpa_cost_modelling import get_resharding_matrix

logger = get_logger(__name__)
logger.setLevel("DEBUG")

import operator
IGNORE_FUNCS = [operator.getitem]
IGNORE_METHODS = [
    "size"
]

def deepgetattr(obj, attr, default=None):
    """Recurses through an attribute chain to get the ultimate value."""
    try:
        return functools.reduce(getattr, attr.split("."), obj)
    except AttributeError:
        return default


def get_node_target(node):
    if isinstance(node.target, str):
        return deepgetattr(node.meta["mase"].model, node.target, None)
    else:
        return node.target


def assign_default_sharding(node):
    rank = len(node.meta["mase"]["common"]["results"]["data_out_0"]["shape"])
    node.meta["mase"]["software"]["autosharding"] = {
        "valid_input_shardings": [{"data_in_0": (SpmdShard.R,) * rank}],
        "valid_output_shardings": [(SpmdShard.R,) * rank],
        "compute_cost_vector": [0],
        "communication_cost_vector": [0],
        "opt_var": np.array([1]),
    }


def mark_choices(mg):
    """
    Once the metadata has already been filled for each op with the possible shardings and costs,
    and the ILP has been solved, this function marks the chosen sharding for each op.
    """
    for node in mg.fx_graph.nodes:
        chosen_idx = (
            0
            if isinstance(
                node.meta["mase"]["software"]["autosharding"]["opt_var"], np.ndarray
            )
            else np.where(
                node.meta["mase"]["software"]["autosharding"]["opt_var"].value == 1
            )[0][0]
        )
        node.meta["mase"]["software"]["autosharding"]["input_sharding"] = node.meta[
            "mase"
        ]["software"]["autosharding"]["valid_input_shardings"][chosen_idx]
        node.meta["mase"]["software"]["autosharding"]["output_sharding"] = node.meta[
            "mase"
        ]["software"]["autosharding"]["valid_output_shardings"][chosen_idx]
        chosen_sharding = {
            key: node.meta["mase"]["software"]["autosharding"]["input_sharding"][key]
            for key in node.meta["mase"]["software"]["autosharding"][
                "input_sharding"
            ].keys()
        }

        # Write into module map (used by distributed launcher)
        target = get_node_target(node)
        if node.op == "call_module" and target is not None:
            module_map[target] = {"node": node.name, "sharding": chosen_sharding}
            module_map[target]["sharding"]["output"] = node.meta["mase"]["software"][
                "autosharding"
            ]["output_sharding"]

    return mg

from torch.distributed._tensor._op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import Replicate, Shard, DTensorSpec
import itertools

def placeholder_or_getattr_strategy(meta, mesh):
    ndims = len(meta["common"]["results"]["data_out_0"]["shape"])
    opts = [Replicate()] + [Shard(dim) for dim in range(ndims)]
    shardings = []
    for sharding in itertools.product(opts, repeat=2):
        spec = DTensorSpec(mesh, sharding)
        shardings.append(PlacementStrategy(
            input_specs=spec,
            output_specs=spec
        ))
    return OpStrategy(shardings)

def alpa_intra_op_sharding_pass(mg, mesh, debug=False):
    """
    Intra-operator auto parallelization pass.
    """

    module_map = {}

    # Setup for the ILP optimization
    expr = 0
    constr = []

    # Write cost vectors into metadata for each operator
    # This will later be used to solve the ILP optimization
    for node in mg.fx_graph.nodes:

        if (node.op == "call_function" and node.target in IGNORE_FUNCS) or (node.op == "call_method" and node.target in IGNORE_METHODS):
            logger.debug(f"Ignoring {node.op} node {node.name} with target {node.target}")
            continue

        # Obtain strategy according to node op
        # ================================================

        if node.op in ["placeholder", "get_attr"]:
            logger.debug(f"Node {node} with op {node.op} will be assigned all permutations of Shard(dims) and Replicate()")
            op_strategy = placeholder_or_getattr_strategy(node.meta["mase"], mesh.mesh_shape)

        elif node.op == "call_method" and node.target in ALPA_METHODS.keys():
            logger.debug(f"Obtaining strategy for node {node.name}")
            op_strategy = ALPA_METHODS[node.target](node.meta["mase"], mesh.mesh_shape)

        elif node.op == "call_function" and node.target in ALPA_FUNCTIONS.keys():
            # Enumerate shardings and costs for this operator
            # (
            #     input_shardings,
            #     output_shardings,
            #     compute_cost_vector,
            #     communication_cost_vector,
            # ) = ALPA_FUNCTIONS[node.target](node.meta, mesh)
            logger.debug(f"Obtaining strategy for node {node.name}")
            op_strategy = ALPA_FUNCTIONS[node.target](node.meta["mase"], mesh.mesh_shape)

        else:
            logger.warning(f"Unknown node {node.name} with op {node.op}")
            continue
            breakpoint()

        # Formulate optimization variable and consider compute/communication cost
        opt_var = cp.Variable(len(op_strategy.strategies), boolean=True)
        constr += [
            cp.sum(opt_var) == 1,
        ]
        # expr += opt_var.T @ (compute_cost_vector + communication_cost_vector)

        # Write into metadata
        node.meta["mase"]["software"]["autosharding"] = {
            "op_strategy": op_strategy,
            "opt_var": opt_var,
        }

        # Consider resharding cost
        # for in_node in node.all_input_nodes:
        #     in_opt_var = in_node.meta["mase"]["software"]["autosharding"]["opt_var"]

        #     resharding_costs = get_resharding_matrix(
        #         mesh,
        #         src_shardings=in_node.meta["mase"]["software"]["autosharding"][
        #             "valid_output_shardings"
        #         ],
        #         dest_shardings=[
        #             sharding["data_in_0"]
        #             for sharding in node.meta["mase"]["software"]["autosharding"][
        #                 "valid_input_shardings"
        #             ]
        #         ],
        #         dest_node_meta=node.meta["mase"],
        #     ).flatten()

        #     # Formulate resharding cost term with linearized variable
        #     e_var = cp.Variable(
        #         opt_var.shape[0] * in_opt_var.shape[0], boolean=True
        #     )
        #     expr += e_var.T @ resharding_costs
        #     constr += [
        #         cp.sum(e_var) == 1,
        #     ]

        #     # Scalar construction of the inequality constraints for the linearized variable
        #     for i in range(e_var.shape[0]):
        #         constr += [
        #             e_var[i] <= opt_var[i // in_opt_var.shape[0]],
        #             e_var[i] <= in_opt_var[i % in_opt_var.shape[0]],
        #             e_var[i]
        #             >= opt_var[i // in_opt_var.shape[0]]
        #             + in_opt_var[i % in_opt_var.shape[0]]
        #             - 1,
        #         ]

    # Solve the ILP problem
    # prob = cp.Problem(cp.Minimize(expr), constr)
    # prob.solve()

    # mg = mark_choices(mg)

    return mg, module_map


def alpa_autosharding_pass(mg, mesh):
    mg, module_map = alpa_intra_op_sharding_pass(mg, mesh)
    return mg, module_map
