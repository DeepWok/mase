import torch
import torch.fx as fx
from torch.distributed._tensor._collective_utils import redistribute_cost
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


logger = get_logger(__name__)
logger.setLevel("DEBUG")

from .mesh_model import MeshModel


def _extract_ilp(mg, mesh, pass_args={}):
    """
    For each node in the graph, assign an OpStrategy object which contains all possible
    sharding algorithms. Also assign opt_var instance which is one-hot vector used to
    solve ILP.

    Return list of constraints associated with ILP. The constraints at this stage only
    enforce that each optimizer variable is a one-hot boolean vector.
    """

    # Setup for the ILP optimization
    constr = []
    expr = 0

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
            op_strategy = placeholder_or_getattr_strategy(
                node.meta["mase"],
                mesh,
                skip_fully_replicated=pass_args.get("skip_fully_replicated", False),
            )

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

        # Formulate optimization variable
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

        # Consider resharding cost for each of the node's arguments
        for arg_idx, in_node in enumerate(node.all_input_nodes):

            # Skip constant nodes
            if not isinstance(in_node, fx.Node) or not isinstance(
                in_node.meta["mase"]["common"]["results"]["data_out_0"]["value"],
                torch.Tensor,
            ):
                continue
            logger.debug(f"Parsing arg {in_node} of node {node}")

            # Fetch this node's input specs
            node_op_strategy = node.meta["mase"]["software"]["autosharding"][
                "op_strategy"
            ]
            node_in_specs = [
                strategy.input_specs[arg_idx]
                for strategy in node_op_strategy.strategies
            ]

            # Fetch the argument node's output specs
            in_opt_var = in_node.meta["mase"]["software"]["autosharding"]["opt_var"]
            arg_op_strategy = in_node.meta["mase"]["software"]["autosharding"][
                "op_strategy"
            ]
            arg_out_specs = [
                strategy.output_specs for strategy in arg_op_strategy.strategies
            ]

            # Formulate resharding cost matrix
            resharding_costs = np.zeros((len(node_in_specs), len(arg_out_specs)))
            for dest_idx, dest_spec in enumerate(node_in_specs):
                for src_idx, src_spec in enumerate(arg_out_specs):
                    cost = redistribute_cost(src_spec, dest_spec)
                    resharding_costs[dest_idx, src_idx] = (
                        1000000 if cost == float("inf") else cost
                    )

            resharding_costs = resharding_costs.flatten()

            # Formulate linearized variable for resharding cost
            e_var = cp.Variable(resharding_costs.shape[0], boolean=True)
            expr += e_var.T @ resharding_costs
            constr += [
                cp.sum(e_var) == 1,
            ]

            # Scalar construction of the inequality constraints for the linearized variable
            for i in range(e_var.shape[0]):
                constr += [
                    e_var[i] <= opt_var[i // in_opt_var.shape[0]],
                    e_var[i] <= in_opt_var[i % in_opt_var.shape[0]],
                    e_var[i]
                    >= opt_var[i // in_opt_var.shape[0]]
                    + in_opt_var[i % in_opt_var.shape[0]]
                    - 1,
                ]

            # Below speeds up compilation but the number of constraints is the same?

            # # Reshape e_var to match the dimensions of opt_var and in_opt_var
            # e_var_reshaped = cp.reshape(e_var, (opt_var.shape[0], in_opt_var.shape[0]))

            # # Create broadcasted versions of opt_var and in_opt_var
            # opt_var_broadcast = cp.reshape(opt_var, (opt_var.shape[0], 1))
            # in_opt_var_broadcast = cp.reshape(in_opt_var, (1, in_opt_var.shape[0]))

            # # Define the vectorized constraints
            # constr += [
            #     e_var_reshaped <= opt_var_broadcast,
            #     e_var_reshaped <= in_opt_var_broadcast,
            #     e_var_reshaped >= opt_var_broadcast + in_opt_var_broadcast - 1,
            # ]

    # Solve the ILP problem
    prob = cp.Problem(cp.Minimize(expr), constr)
    return mg, prob


def _export_solution(mg):

    nodes = [node for node in mg.fx_graph.nodes]
    node_names = [node.name for node in nodes]
    opt_vars = [
        node.meta["mase"]["software"]["autosharding"]["opt_var"] for node in nodes
    ]
    opt_vals = [i.value if i is not None else None for i in opt_vars]
    choices = [np.argmax(i) for i in opt_vals]

    strategies = [
        i.meta["mase"]["software"]["autosharding"]["op_strategy"].strategies
        for i in nodes
    ]
    shardings = [strat[choices[idx]] for idx, strat in enumerate(strategies)]
    map = [
        {
            "node": nodes[idx].name,
            "input_specs": strat.input_specs,
            "output_specs": strat.output_specs,
        }
        for idx, strat in enumerate(shardings)
    ]

    breakpoint()

    return mg, {}


def _mark_sharding(mg):
    for node in mg.fx_graph.nodes:
        opt_var = node.meta["mase"]["software"]["autosharding"]["opt_var"]

        if opt_var is None:
            continue

        idx = np.where(opt_var.value == 1)

    return mg, {}


def alpa_intra_op_sharding_pass(mg, mesh, pass_args={}, debug=False):
    """Intra-operator auto parallelization pass from the Alpa paper: https://arxiv.org/abs/2201.12023

    Args:
        mg (MaseGraph): Input MaseGraph.
        mesh (MeshModel): mesh description.
        pass_args (dict, optional): pass arguments. Defaults to {}.
        debug (bool, optional): enable debug. Defaults to False.

    Returns:
        MaseGraph: annotated MaseGraph.
    """

    module_map = {}

    # Formulate and solve the ILP
    logger.info(f"Formulating the ILP...")
    mg, problem = _extract_ilp(mg, mesh, pass_args)

    logger.info(f"Solving the ILP...")
    problem.solve(verbose=True, scipy_options={"disp": True})

    mg, _ = _export_solution(mg)
    mg, _ = _mark_sharding(mg)

    return mg, module_map
