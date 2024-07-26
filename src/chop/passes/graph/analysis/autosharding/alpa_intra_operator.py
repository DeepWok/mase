import math
import numpy as np
import cvxpy as cp
from copy import copy

import torch
import torch.fx as fx
from torch.distributed._tensor._collective_utils import redistribute_cost
from torch.distributed._tensor._op_schema import DTensorSpec, OpStrategy
from torch.distributed._tensor.placement_types import Shard

from chop.tools import get_logger
from chop.tools.utils import deepgetattr
from .mesh_model import MeshModel

from .layers import (
    AUTOSHARDING_MODULES,
    AUTOSHARDING_FUNCTIONS,
    AUTOSHARDING_METHODS,
    IMPLICIT_FUNCS,
    IMPLICIT_METHODS,
)
from .strategies.common import (
    fully_replicated_strategy,
    placeholder_or_getattr_strategy,
)


logger = get_logger(__name__)
logger.setLevel("DEBUG")


def _get_computation_cost_from_strategy(
    node: fx.Node,
    strategy: OpStrategy,
    mesh: MeshModel,
    repeat: int = 5,
    warmup_iters: int = 2,
    profiling_device: int = 0,
):
    """
    ...

    Args:
        node (fx.Node): _description_
        strategy (OpStrategy): _description_
        repeat (int, optional): _description_. Defaults to 5.
        warmup_iters (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    arg_specs = strategy.input_specs
    arg_specs = [arg_specs] if isinstance(arg_specs, DTensorSpec) else arg_specs

    # Formulate list of arguments to run the target with
    args = []
    for arg_idx, arg_spec in enumerate(arg_specs):

        # If tensor meta is None, this is not a sharded argument
        if arg_spec.tensor_meta is None:
            key = list(node.meta["mase"]["common"]["args"].keys())[arg_idx]
            arg_value = node.meta["mase"]["common"]["args"][key]["value"]
            args.append(arg_value)
            continue

        # If it is a sharded argument, find the local tensor shape
        else:
            global_shape = copy(arg_spec.tensor_meta.shape)
            local_shape = copy(arg_spec.tensor_meta.shape)

            # Check if each tensor dimension is sharded to update local_shape
            for dim in range(len(global_shape)):
                # Get device mesh dimensions along which dimension 'dim' of the tensor is sharded
                sharded_mesh_dims = [
                    idx
                    for idx, plac in enumerate(arg_spec.placements)
                    if plac == Shard(dim)
                ]

                # This tensor dimension is not sharded
                if len(sharded_mesh_dims) == 0:
                    continue

                # This tensor dimension is fully sharded
                elif len(sharded_mesh_dims) == 2:
                    num_gpus = np.prod(mesh.mesh_shape)

                # This tensor dimension is sharded along one mesh dimension
                elif len(sharded_mesh_dims) == 1:
                    num_gpus = mesh.mesh_shape[sharded_mesh_dims[0]]

                # Define the local shape with minimum == 1
                local_shape[dim] = math.ceil(global_shape[dim] / num_gpus)

            # Generate a random tensor with the local shape
            args.append(
                torch.randn(
                    local_shape,
                    device=f"cuda:{profiling_device}",
                )
            )

    # Get target function
    fn = node.target

    # Run the function with the arguments
    start_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(repeat)
    ]
    end_event = [
        torch.cuda.Event(enable_timing=True, blocking=True) for _ in range(repeat)
    ]

    torch.cuda.empty_cache()

    for idx in range(repeat):
        start_event[idx].record()
        _ = fn(*args)
        end_event[idx].record()
    torch.cuda.synchronize()

    elapsed = [start_event[idx].elapsed_time(end_event[idx]) for idx in range(repeat)]

    return np.mean(elapsed[warmup_iters:])


def _extract_ilp(mg, mesh, pass_args={}):
    """
    For each node in the graph, assign an OpStrategy object which contains all possible
    sharding algorithms. Also assign opt_var instance which is one-hot vector used to
    solve ILP.

    Return list of constraints associated with ILP. The constraints at this stage only
    enforce that each optimizer variable is a one-hot boolean vector.

    Args:
        mg (MaseGraph): input mase graph.
        mesh (MeshModel): mesh model.
        pass_args (dict, optional): pass arguments. Defaults to {}.

    Returns:
        MaseGraph: input mase graph.
        cp.Problem: optimization problem.
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
            opt_var = cp.Variable(1, boolean=True)
            constr += [
                cp.sum(opt_var) == 1,
            ]

            # Opt var is None since no decision needs to be taken
            node.meta["mase"]["software"]["autosharding"] = {
                "op_strategy": op_strategy,
                "opt_var": opt_var,
            }
            continue

        # Output nodes simply propagate op_strategy from their input nodes
        elif node.op == "output":
            logger.debug(
                f"Op strategy from node {node.all_input_nodes[0]} is propagated to {node} node."
            )
            opt_var = cp.Variable(1, boolean=True)
            constr += [
                cp.sum(opt_var) == 1,
            ]
            node.meta["mase"]["software"]["autosharding"] = {
                "op_strategy": node.all_input_nodes[0].meta["mase"]["software"][
                    "autosharding"
                ]["op_strategy"],
                "opt_var": opt_var,
            }
            continue

        # Obtain strategy according to node op
        # ================================================

        elif node.op in [
            "placeholder",
            "get_attr",
        ]:
            logger.debug(
                f"Node {node} with op {node.op} will be assigned all permutations of Shard(dims) and Replicate()"
            )
            op_strategy = placeholder_or_getattr_strategy(
                node.meta["mase"],
                mesh,
                skip_fully_replicated=pass_args.get("skip_fully_replicated", False),
            )

        elif (
            node.op == "call_function" and node.target in AUTOSHARDING_FUNCTIONS.keys()
        ):
            logger.debug(f"Obtaining strategy for call_function node: {node.name}")
            op_strategy = AUTOSHARDING_FUNCTIONS[node.target](node.meta["mase"], mesh)

        else:
            logger.warning(
                f"Unknown node {node.name} with op {node.op} with be allocated fully replicated strategy."
            )
            op_strategy = fully_replicated_strategy(node.meta["mase"], mesh)
            opt_var = cp.Variable(1, boolean=True)
            constr += [
                cp.sum(opt_var) == 1,
            ]
            node.meta["mase"]["software"]["autosharding"] = {
                "op_strategy": fully_replicated_strategy(
                    node.meta["mase"],
                    mesh,
                ),
                "opt_var": opt_var,
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
        }

        # Consider computation cost (c_v term) for each of the node's strategies
        # placeholder/get_attr/output nodes have no computation cost
        if node.op not in [
            "placeholder",
            "get_attr",
            "output",
            # todo: decide how to handle call_method nodes
            "call_method",
        ]:
            cost_vector = []
            try:
                for strategy in op_strategy.strategies:
                    cost = _get_computation_cost_from_strategy(node, strategy, mesh)
                    cost_vector.append(cost)
            except:
                print(f"Op {node} failed to compute cost")

        # Consider resharding cost for each of the node's arguments
        e_var_checks = []
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
                (
                    [strategy.input_specs][arg_idx]
                    if isinstance(strategy.input_specs, DTensorSpec)
                    else strategy.input_specs[arg_idx]
                )
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
            resharding_costs = np.zeros((opt_var.shape[0], in_opt_var.shape[0]))

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

            # After solving the ILP, verify constraints were correctly formulated
            if pass_args.get("run_checks", False):
                e_var_checks.append((opt_var, in_opt_var, e_var))

            # Constraints s.t. e_var = outer(opt_var, in_opt_var)
            indices = np.arange(e_var.shape[0])
            opt_indices, in_opt_indices = np.divmod(indices, in_opt_var.shape[0])
            constr += [
                e_var <= opt_var[opt_indices],
                e_var <= in_opt_var[in_opt_indices],
                e_var >= opt_var[opt_indices] + in_opt_var[in_opt_indices] - 1,
            ]

        if pass_args.get("run_checks", False):
            node.meta["mase"]["software"]["autosharding"]["e_var_checks"] = e_var_checks

    # Solve the ILP problem
    prob = cp.Problem(cp.Minimize(expr), constr)
    return mg, prob


def _run_checks(mg, pass_args):
    """
    Run checks on the ILP solution to ensure that the constraints were correctly formulated.

    Args:
        mg (MaseGraph): input mase graph.
        pass_args (dict): pass arguments.

    Returns:
        None
    """

    for node in mg.fx_graph.nodes:
        check_list = node.meta["mase"]["software"]["autosharding"].get(
            "e_var_checks", []
        )

        # Check that the constraints on the linearised variable for resharding cost
        # are correctly formulated
        for opt_var, in_opt_var, e_var in check_list:
            idx1 = np.where(opt_var.value == 1)[0][0]
            idx2 = np.where(in_opt_var.value == 1)[0][0]
            idx3 = np.where(e_var.value == 1)[0][0]
            assert (
                idx3 == idx1 * in_opt_var.shape[0] + idx2
            ), f"Linearized variable for resharding cost is not consistent for node {node}."


def _mark_sharding(mg, pass_args):
    """
    After solving the ILP, annotate the metadata of each operator in the graph with the chosen
    parallelization strategy.

    Args:
        mg (MaseGraph): input mase graph.
        pass_args (dict): pass arguments.

    Returns:
        MaseGraph: input mase graph.
        dict: tensor sharding map.
    """

    logger.info(
        f"Autosharding optimization finished, annotating graph with chosen sharding strategies for each node."
    )

    for node in mg.fx_graph.nodes:
        opt_var = node.meta["mase"]["software"]["autosharding"]["opt_var"]

        if opt_var is None:
            chosen_strategy = node.meta["mase"]["software"]["autosharding"][
                "op_strategy"
            ]
        else:
            # Get the strategy chosen by the ILP
            try:
                idx = np.where(opt_var.value == 1)[0][0]
            except:
                idx = np.argmax(opt_var.value)

            chosen_strategy = node.meta["mase"]["software"]["autosharding"][
                "op_strategy"
            ].strategies[idx]

        # Annotate chosen placement strategy
        node.meta["mase"]["software"]["autosharding"][
            "placement_strategy"
        ] = chosen_strategy

        # Annotate arg metadata with chosen strategy
        arg_specs = chosen_strategy.input_specs
        if isinstance(arg_specs, DTensorSpec):
            arg_specs = (arg_specs,)

        if not node.op in ["placeholder", "get_attr", "output"]:
            assert len(arg_specs) == len(
                node.meta["mase"]["common"]["args"].keys()
            ), "Number of arguments do not match metadata."

        out_spec = chosen_strategy.output_specs

        if node.op in ["placeholder", "get_attr", "output"]:
            node.meta["mase"]["common"]["results"]["data_out_0"][
                "dtensor_spec"
            ] = out_spec

        # call_function nodes
        else:
            arg_list = [i for i in node.meta["mase"]["common"]["args"].keys()]

            for arg_idx, arg_spec in enumerate(arg_specs):
                arg_meta = node.meta["mase"]["common"]["args"][arg_list[arg_idx]]
                if not isinstance(arg_meta, dict):
                    continue
                arg_meta["dtensor_spec"] = arg_spec

        # Annotate output metadata with chosen strategy
        node.meta["mase"]["common"]["results"]["data_out_0"]["dtensor_spec"] = out_spec

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

    # Formulate and solve the ILP
    logger.info(f"Formulating the ILP...")
    mg, problem = _extract_ilp(mg, mesh, pass_args)

    logger.info(f"Solving the ILP...")
    problem.solve(
        verbose=True,
        scipy_options={
            "disp": True,
            "time_limit": pass_args.get("time_limit", None),
            "mip_rel_gap": pass_args.get("mip_rel_gap", 0) / 100,
        },
    )

    if pass_args.get("run_checks", False):
        _run_checks(mg, pass_args)

    mg, _ = _mark_sharding(mg, pass_args)

    return mg, {"solution": problem.value}
