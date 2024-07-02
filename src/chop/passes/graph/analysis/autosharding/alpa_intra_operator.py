import sys, pdb, traceback
import functools

import torch.nn as nn
import numpy as np
import cvxpy as cp

from chop.tools import get_logger

from .layers import (
    AUTOSHARDING_FUNCTIONS,
    AUTOSHARDING_METHODS,
    IMPLICIT_FUNCS,
    IMPLICIT_METHODS,
    placeholder_or_getattr_strategy,
    fully_replicated_strategy,
)
from .alpa_cost_modelling import get_resharding_matrix

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def _enumerate_sharding_strategies(mg, mesh):
    """
    For each node in the graph, assign an OpStrategy object which contains all possible
    sharding algorithms. Also assign opt_var instance which is one-hot vector used to
    solve ILP.

    Return list of constraints associated with ILP. The constraints at this stage only
    enforce that each optimizer variable is a one-hot boolean vector.
    """

    # Setup for the ILP optimization
    constr = []

    # Find sharding strategies for each operator in the graph
    for node in mg.fx_graph.nodes:

        if (node.op == "call_function" and node.target in IMPLICIT_FUNCS) or (
            node.op == "call_method" and node.target in IMPLICIT_METHODS
        ):
            logger.debug(
                f"Implicit {node.op} node {node.name} was assigned fully replicated sharding."
            )

            op_strategy = fully_replicated_strategy(node.meta["mase"], mesh)

            # Opt var is None since no decision needs to be taken
            node.meta["mase"]["software"]["autosharding"] = {
                "op_strategy": op_strategy,
                "opt_var": None,
                "input": None,
                "output": None,
            }
            continue

        # Obtain strategy according to node op
        # ================================================

        if node.op in ["placeholder", "get_attr"]:
            logger.debug(
                f"Node {node} with op {node.op} will be assigned all permutations of Shard(dims) and Replicate()"
            )
            op_strategy = placeholder_or_getattr_strategy(node.meta["mase"], mesh)

        elif node.op == "output":
            logger.debug(
                f"Op strategy from node {node.args[0]} is propagated to {node} node."
            )
            node.meta["mase"]["software"]["autosharding"] = {
                "op_strategy": node.args[0].meta["mase"]["software"]["autosharding"][
                    "op_strategy"
                ],
                "opt_var": None,
                "input": None,
                "output": None,
            }
            continue

        elif node.op == "call_method" and node.target in AUTOSHARDING_METHODS.keys():
            logger.debug(f"Obtaining strategy for node {node.name}")
            op_strategy = AUTOSHARDING_METHODS[node.target](node.meta["mase"], mesh)

        elif (
            node.op == "call_function" and node.target in AUTOSHARDING_FUNCTIONS.keys()
        ):
            logger.debug(f"Obtaining strategy for node {node.name}")
            op_strategy = AUTOSHARDING_FUNCTIONS[node.target](node.meta["mase"], mesh)

        else:
            logger.warning(f"Unknown node {node.name} with op {node.op}")
            node.meta["mase"]["software"]["autosharding"] = {
                "op_strategy": fully_replicated_strategy(node.meta["mase"], mesh),
                "opt_var": None,
                "input": None,
                "output": None,
            }
            continue

        # Formulate optimization variable and consider compute/communication cost
        opt_var = cp.Variable(len(op_strategy.strategies), boolean=True)
        constr += [
            cp.sum(opt_var) == 1,
        ]

        # Write into metadata
        node.meta["mase"]["software"]["autosharding"] = {
            "op_strategy": op_strategy,
            "opt_var": opt_var,
            "input": None,
            "output": None,
        }

        import torch
        import torch.fx as fx
        from torch.distributed._tensor._collective_utils import redistribute_cost

        for arg_idx, in_node in enumerate(node.all_input_nodes):
            if not isinstance(in_node, fx.Node) or not isinstance(
                in_node.meta["mase"]["common"]["results"]["data_out_0"]["value"],
                torch.Tensor,
            ):
                continue
            print(f"Parsing arg {in_node} of node {node}")

            node_op_strategy = node.meta["mase"]["software"]["autosharding"][
                "op_strategy"
            ]
            arg_op_strategy = in_node.meta["mase"]["software"]["autosharding"][
                "op_strategy"
            ]

            arg_out_specs = [
                strategy.output_specs for strategy in arg_op_strategy.strategies
            ]
            node_in_specs = [
                strategy.input_specs[arg_idx]
                for strategy in node_op_strategy.strategies
            ]

            for out_spec in arg_out_specs:
                for in_spec in node_in_specs:
                    cost = redistribute_cost(out_spec, in_spec)
                    # print(
                    #     f"Cost for {out_spec} -> {in_spec}: {cost}"
                    # )

    return mg, constr


def alpa_intra_op_sharding_pass(mg, mesh, debug=False):
    """
    Intra-operator auto parallelization pass.
    """

    module_map = {}

    mg, constr = _enumerate_sharding_strategies(mg, mesh)

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
