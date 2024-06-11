import numpy as np
import cvxpy as cp

from chop.tools import get_logger

from .common import Shard
from .alpa_layers import ALPA_LAYERS
from .alpa_cost_modelling import get_resharding_matrix

logger = get_logger(__name__)

def alpa_intra_op_sharding_pass(mg, mesh):
    """
    Intra-operator auto parallelization pass.
    """

    # Setup for the ILP optimization
    expr = 0
    constr = []

    # Write cost vectors into metadata for each operator
    # This will later be used to solve the ILP optimization
    for node in mg.fx_graph.nodes:

        # Extract the target
        if isinstance(node.target, str):
            target = getattr(node.meta["mase"].model, node.target, None)
            target_cls = type(target)
        else:
            target = node.target

        if target_cls in ALPA_LAYERS.keys():
            # Enumerate shardings and costs for this operator
            (
                input_shardings,
                output_shardings,
                compute_cost_vector,
                communication_cost_vector,
            ) = ALPA_LAYERS[target_cls](node.meta, mesh)

            # Formulate optimization variable and consider compute/communication cost
            opt_var = cp.Variable(len(input_shardings), boolean=True)
            constr += [
                cp.sum(opt_var) == 1,
            ]
            expr += opt_var.T @ (compute_cost_vector + communication_cost_vector)

            # Write into metadata
            node.meta["mase"]["software"]["autosharding"] = {
                "valid_input_shardings": input_shardings,
                "valid_output_shardings": output_shardings,
                "compute_cost_vector": compute_cost_vector,
                "communication_cost_vector": communication_cost_vector,
                "opt_var": opt_var,
            }

            # Consider resharding cost
            for in_node in node.all_input_nodes:
                in_opt_var = in_node.meta["mase"]["software"]["autosharding"]["opt_var"]

                resharding_costs = get_resharding_matrix(
                    mesh,
                    src_shardings = in_node.meta["mase"]["software"]["autosharding"]["valid_output_shardings"], 
                    dest_shardings = [sharding[0] for sharding in node.meta["mase"]["software"]["autosharding"]["valid_input_shardings"]],
                    dest_node_meta = node.meta["mase"]
                ).flatten()

                # Formulate resharding cost term with linearized variable
                e_var = cp.Variable(opt_var.shape[0] * in_opt_var.shape[0], boolean=True)
                expr += e_var.T @ resharding_costs
                constr += [
                    cp.sum(e_var) == 1,
                ]

                # Scalar construction of the inequality constraints for the linearized variable
                for i in range(e_var.shape[0]):
                    constr += [
                        e_var[i] <= opt_var[i // in_opt_var.shape[0]],
                        e_var[i] <= in_opt_var[i % in_opt_var.shape[0]],
                        e_var[i] >= opt_var[i // in_opt_var.shape[0]] + in_opt_var[i % in_opt_var.shape[0]] - 1
                    ]

        # Inputs to the whole graph are always replicated across all devices
        elif node.op == "placeholder" or node.op == "output":
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

    # Solve the ILP problem
    prob = cp.Problem(cp.Minimize(expr), constr)
    prob.solve()

    for node in mg.fx_graph.nodes:
        if isinstance(node.meta["mase"]["software"]["autosharding"]["opt_var"], np.ndarray):
            chosen_idx = 0
        else:
            chosen_idx = np.where(node.meta["mase"]["software"]["autosharding"]["opt_var"].value == 1)[0][0]
        node.meta["mase"]["software"]["autosharding"]["input_sharding"] = node.meta["mase"]["software"]["autosharding"]["valid_input_shardings"][chosen_idx]
        node.meta["mase"]["software"]["autosharding"]["output_sharding"] = node.meta["mase"]["software"]["autosharding"]["valid_output_shardings"][chosen_idx]

    return mg, {}

def alpa_autosharding_pass(mg, mesh):
    return alpa_intra_op_sharding_pass(mg, mesh)