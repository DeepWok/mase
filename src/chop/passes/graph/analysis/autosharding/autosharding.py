import numpy as np
import cvxpy as cp
from time import time
import csv
import dill

from torch.distributed._tensor._op_schema import DTensorSpec
from torch.distributed._tensor.placement_types import Replicate

from chop.tools import get_logger

from .mesh_model import MeshModel
from .alpa import alpa_autosharding_pass

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def deepgetattr(obj, attr, default=None):
    """Recurses through an attribute chain to get the ultimate value."""
    import functools

    try:
        return functools.reduce(getattr, attr.split("."), obj)
    except AttributeError:
        return default


def _import_solution(mg, solution: dict, mesh: MeshModel):
    """Import an autosharding solution into the metadata of the MaseGraph.

    Args:
        mg (MaseGraph): input mase graph.
        solution (dict): autosharding solution.

    Returns:
        MaseGraph: input mase graph.
        dict: empty dictionary.
    """
    for node in mg.fx_graph.nodes:
        if node.name not in solution.keys():
            continue

        for arg, arg_spec in solution[node.name].get("args", {}).items():
            node.meta["mase"]["common"]["args"][arg]["dtensor_spec"] = DTensorSpec(
                mesh=mesh, placements=arg_spec
            )

        for result, result_spec in solution[node.name].get("results", {}).items():
            node.meta["mase"]["common"]["results"][result]["dtensor_spec"] = (
                DTensorSpec(mesh=mesh, placements=result_spec)
            )

    return mg, {}


def _export_solution(mg, export_file: str = "ilp_solution.csv"):
    """Export the ILP solution to a csv file.

    Args:
        mg (MaseGraph): input mase graph.
        export_file (str, optional): output file name. Defaults to "ilp_solution.csv".

    Returns:
        MaseGraph: input mase graph.
        dict: empty dictionary.
    """
    # Reduce metadata to autosharding solution
    out_dict = {}
    for node in mg.fx_graph.nodes:
        node_name = node.name
        out_dict[node_name] = {
            "args": {},
            "results": {},
        }
        for arg, arg_info in node.meta["mase"]["common"]["args"].items():
            if not isinstance(arg_info, dict):
                continue
            out_dict[node_name]["args"][arg] = arg_info.get(
                "dtensor_spec", DTensorSpec(None, (Replicate(), Replicate()))
            ).placements

        for result, result_info in node.meta["mase"]["common"]["results"].items():
            if not isinstance(result_info, dict):
                continue
            out_dict[node_name]["results"][result] = result_info.get(
                "dtensor_spec", DTensorSpec(None, (Replicate(), Replicate()))
            ).placements

    with open(export_file.replace(".csv", ".pkl"), "wb") as file:
        dill.dump(out_dict, file)

    return mg, {}


def _get_sharding_map(mg):
    """
    Export the tensor sharding map to a dictionary, to be used by the MaseLauncher for
    distributed deployment.

    Args:
        mg (MaseGraph): input mase graph.

    Returns:
        MaseGraph: input mase graph.
        dict: tensor sharding map.

    The tensor sharding map is a dictionary with the following structure.
    {
        module: {
            node: node_name,
            sharding: {
                attr: out_specs,
            },
        },
    }
    """

    tensor_sharding_map = {}
    for node in mg.fx_graph.nodes:
        if node.op == "get_attr":
            module_str = ".".join(node.target.split(".")[:-1])
            attr = node.target.split(".")[-1]
            module = deepgetattr(node.meta["mase"].model, module_str)

            out_specs = node.meta["mase"]["common"]["results"]["data_out_0"][
                "dtensor_spec"
            ]

            if module not in tensor_sharding_map:
                tensor_sharding_map[module] = {
                    "node": node.name,
                    "sharding": {
                        attr: out_specs,
                    },
                }
            else:
                tensor_sharding_map[module]["sharding"][attr] = out_specs

    return tensor_sharding_map


def autosharding_analysis_pass(mg, pass_args: dict = {}):
    """Annotate the metadata of each operator in the graph with a parallelization strategy.

    Args:
        mg (MaseGraph): input mase graph.
        pass_args (dict, optional): pass arguments. Defaults to {}.

    Returns:
        MaseGraph: annotated mase graph.

    The pass_args dictionary expects the following elements.

    - mesh_shape -> tuple : Shape of the device cluster. Should be a 2-dimensional tuple.
    - inter_node_bandwidth -> int : Inter-node bandwidth, i.e. between GPU nodes.
    - intra_node_bandwidth -> int : Intra-node bandwidth, i.e. between GPU devices in each node.

    Additionally, the following elements can be passed.

    - algo (optional) -> str : Sharding algorithm to use. Default is "alpa".
    - communications_backend (optional) -> str : Communications backend to use, e.g. "nccl" or "gloo". Default is "nccl".
    - skip_fully_replicated (optional) -> bool : If set to true, do not consider fully replicated sharding as an option for any operator.
    - time_limit (optional) -> int : Time limit for the ILP solver, in seconds. Default is 10000.
    - mip_rel_gap (optional) -> int : MIP relative gap for the ILP solver. Default is 0 (i.e. obtain full solution).
    - run_checks (optional) -> bool : If set to true, run checks on the autosharding solution. Default is False.
    - preload_solution (optional) -> bool : If set to true, preload autosharding solution from file.
    - ilp_solution_file (optional) -> str : File to export the autosharding solution to. Defaults to: "ilp_solution.pkl".
    """

    assert (
        "mesh_shape" in pass_args
    ), "Logical description for device cluster was not specified."
    assert "inter_node_bandwidth" in pass_args, "Inter-node bandwidth not specified"
    assert "intra_node_bandwidth" in pass_args, "Intra-node bandwidth not specified"

    # Initialize device mesh model, used for cost estimation
    mesh = MeshModel(pass_args["mesh_shape"])

    # Preload autosharding solution
    if pass_args.get("preload_solution", False):
        fname = pass_args.get("ilp_solution_file", "ilp_solution.pkl")
        logger.info(f"Preloading autosharding solution from: {fname}")
        with open(fname, "rb") as file:
            solution = dill.load(file)

        # Annotate the metadata of each operator with the autosharding solution
        mg, pass_outs = _import_solution(mg, solution, mesh)
        autosharding_time = 0

    # Run autosharding pass
    else:
        # Define autosharding backend
        algo = pass_args.get("sharding_algo", "alpa")

        # Communication cost model depends
        mesh.set_cost_model_parameters(
            intra_node_bandwidth=pass_args["intra_node_bandwidth"],
            inter_node_bandwidth=pass_args["inter_node_bandwidth"],
            backend=pass_args.get("communications_backend", "default"),
        )

        # Run intra-operator pass
        start_time = time()
        if algo == "alpa":
            mg, pass_outs = alpa_autosharding_pass(mg, mesh, pass_args)

        end_time = time()
        autosharding_time = end_time - start_time
        logger.info(
            f"Autosharding pass complete. Time taken: {autosharding_time} seconds. Solution: {pass_outs['solution']}"
        )

        # Export solution
        fname = pass_args.get("ilp_solution_file", "ilp_solution.pkl")
        logger.info(f"Exporting solution to {fname}")
        mg, _ = _export_solution(mg, export_file=fname)

    if not pass_args.get(f"skip_forward", False):
        tensor_sharding_map = _get_sharding_map(mg)

    return mg, {
        "autosharding_time": autosharding_time,
        "tensor_sharding_map": tensor_sharding_map,
        **pass_outs,
    }
