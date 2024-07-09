import numpy as np
import cvxpy as cp
from time import time

from chop.tools import get_logger

from .mesh_model import MeshModel
from .alpa import alpa_autosharding_pass

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def autosharding_analysis_pass(mg, pass_args: dict = {}):
    """Annotate the metadata of each operator in the graph with a parallelization strategy.

    Args:
        mg (MaseGraph): input mase graph.
        pass_args (dict, optional): pass arguments. Defaults to {}.

    Returns:
        MaseGraph: annotated mase graph.

    The pass_args dictionary expects the following elements.

    - algo (optional) -> str : Sharding algorithm to use. Default is "alpa".
    - mesh_shape -> tuple : Shape of the device cluster. Should be a 2-dimensional tuple.
    - inter_node_bandwidth -> int : Inter-node bandwidth, i.e. between GPU nodes.
    - intra_node_bandwidth -> int : Intra-node bandwidth, i.e. between GPU devices in each node.

    Additionally, the following elements can be passed.

    - communications_backend (optional) -> str : Communications backend to use, e.g. "nccl" or "gloo". Default is "nccl".
    - skip_fully_replicated (optional) -> bool : If set to true, do not consider fully replicated sharding as an option for any operator.
    """

    assert (
        "mesh_shape" in pass_args
    ), "Logical description for device cluster was not specified."
    assert "inter_node_bandwidth" in pass_args, "Inter-node bandwidth not specified"
    assert "intra_node_bandwidth" in pass_args, "Intra-node bandwidth not specified"

    # Timing
    start_time = time()

    # Initialize device mesh model, used for cost estimation
    mesh = MeshModel(pass_args["mesh_shape"])

    algo = pass_args.get("sharding_algo", "alpa")

    # Communication cost model depends
    mesh.set_cost_model_parameters(
        intra_node_bandwidth=pass_args["intra_node_bandwidth"],
        inter_node_bandwidth=pass_args["inter_node_bandwidth"],
        backend=pass_args.get("communications_backend", "default"),
    )

    # Run intra-operator pass
    if algo == "alpa":
        mg, pass_outs = alpa_autosharding_pass(mg, mesh, pass_args)

    end_time = time()
    time_taken = end_time - start_time
    logger.info(
        f"Autosharding pass complete. Time taken: {time_taken} seconds. Solution: {pass_outs['solution']}"
    )

    return mg, {"autosharding_time": time_taken, **pass_outs}
