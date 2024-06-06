import numpy as np
import cvxpy as cp
import pulp

from chop.tools import get_logger

from .autosharding_layers import SHARDING_ALGOS, Shard

logger = get_logger(__name__)


def alpa_intra_op_sharding_pass(mg):
    """
    Intra-operator auto parallelization pass.
    """

    # Setup for the ILP optimization
    expr = 0
    constr = []
    variables = []

    # Write cost vectors into metadata for each operator
    # This will later be used to solve the ILP optimization
    for node in mg.fx_graph.nodes:

        # Extract the target
        if isinstance(node.target, str):
            target = getattr(node.meta["mase"].model, node.target, None)
            target_cls = type(target)
        else:
            target = node.target

        if target_cls in SHARDING_ALGOS.keys():
            # Enumerate shardings and costs for this operator
            (
                input_shardings,
                output_shardings,
                compute_cost_vector,
                communication_cost_vector,
            ) = SHARDING_ALGOS[target_cls]()

            # Formulate optimization variables
            num_shardings = len(input_shardings)
            opt_var = cp.Variable(num_shardings, boolean=True)
            variables.append(opt_var)

            # Constrain choice to be a onehot vector
            constr += [
                cp.sum(opt_var) == 1,
            ]

            # Consider compute and communication cost
            cost_sum = np.array(compute_cost_vector) + np.array(
                communication_cost_vector
            )
            expr += variables[-1].T @ (cost_sum)

            # Consider resharding cost
            for in_node in node.all_input_nodes:
                in_opt_var = in_node.meta["mase"]["software"]["autosharding"]["opt_var"]
                resharding_costs = np.random.randint(
                    1, 10, size=(opt_var.shape + in_opt_var.shape)
                )
                flattened_resharding_cost = np.matrix.flatten(resharding_costs)

                e_var = cp.Variable(opt_var.shape + in_opt_var.shape, boolean=True)
                expr += cp.vec(e_var).T @ flattened_resharding_cost

                constr += [
                    cp.sum(e_var) == 1,
                    # e_var == cp.vec(cp.outer(opt_var, in_opt_var)),
                ]

            # Write into metadata
            node.meta["mase"]["software"]["autosharding"] = {
                "valid_input_shardings": input_shardings,
                "valid_output_shardings": output_shardings,
                "compute_cost_vector": compute_cost_vector,
                "communication_cost_vector": communication_cost_vector,
                "opt_var": opt_var,
            }

        elif node.op == "placeholder":
            # Inputs to the whole graph are always replicated across all devices
            rank = len(node.meta["mase"]["common"]["results"]["data_out_0"]["shape"])
            node.meta["mase"]["software"]["autosharding"] = {
                "valid_input_shardings": [(Shard.R,) * rank],
                "valid_output_shardings": [(Shard.R,) * rank],
                "compute_cost_vector": [0],
                "communication_cost_vector": [0],
                "opt_var": np.array([1]),
            }

        else:
            logger.warning(f"No sharding algorithm found for operator: {target_cls}")

    # Solve the ILP optimization
    prob = cp.Problem(cp.Minimize(expr), constr)
    prob.solve()

    breakpoint()

    return mg, {}


def autosharding_analysis_pass(mg):
    """
    A lightweight implementation of the core algorithm from the Alpa paper: https://arxiv.org/abs/2201.12023
    """

    mg, _ = alpa_intra_op_sharding_pass(mg)

    return mg, {}
