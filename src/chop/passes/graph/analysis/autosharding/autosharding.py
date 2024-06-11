
import torch
import numpy as np
import cvxpy as cp

from chop.tools import get_logger

from .mesh import Mesh
from .autosharding_layers import SHARDING_ALGOS, Shard
from .cost_modelling import get_resharding_matrix

logger = get_logger(__name__)


def alpa_intra_op_sharding_pass(mg, mesh):
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
            ) = SHARDING_ALGOS[target_cls](node.meta, mesh)


            # Formulate optimization variables
            num_shardings = len(input_shardings)
            opt_var = cp.Variable(num_shardings, boolean=True)
            variables.append(opt_var)

            # Write into metadata
            node.meta["mase"]["software"]["autosharding"] = {
                "valid_input_shardings": input_shardings,
                "valid_output_shardings": output_shardings,
                "compute_cost_vector": compute_cost_vector,
                "communication_cost_vector": communication_cost_vector,
                "opt_var": opt_var,
            }

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

                resharding_costs = get_resharding_matrix(
                    mesh,
                    src_shardings = in_node.meta["mase"]["software"]["autosharding"]["valid_output_shardings"], 
                    dest_shardings = [sharding[0] for sharding in node.meta["mase"]["software"]["autosharding"]["valid_input_shardings"]],
                    dest_node_meta = node.meta["mase"]
                ).flatten()

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

        elif node.op == "placeholder" or node.op == "output":
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

    return mg, {}


def autosharding_analysis_pass(mg, pass_args: dict = {}):
    """
    A lightweight implementation of the core algorithm from the Alpa paper: https://arxiv.org/abs/2201.12023
    """

    assert "mesh_shape" in pass_args, "Logical description for device cluster was not specified."
    assert "inter_node_bandwidth" in pass_args, "Inter-node bandwidth not specified"
    assert "intra_node_bandwidth" in pass_args, "Intra-node bandwidth not specified"

    # Initialize representation of device mesh, used for cost estimation
    mesh = Mesh(pass_args["mesh_shape"])

    # Communication cost model depends
    mesh.set_cost_model_parameters(
        intra_node_bandwidth=pass_args["intra_node_bandwidth"],
        inter_node_bandwidth=pass_args["inter_node_bandwidth"],
        backend = pass_args.get("communications_backend", "default")
    )

    # Run intra-operator pass
    mg, _ = alpa_intra_op_sharding_pass(mg, mesh)

    return mg, {}
