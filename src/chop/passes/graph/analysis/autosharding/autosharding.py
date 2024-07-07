import numpy as np
import cvxpy as cp
from time import time

from chop.tools import get_logger

from .mesh_model import MeshModel
from .alpa import alpa_autosharding_pass

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def autosharding_analysis_pass(mg, pass_args: dict = {}):
    """
    A lightweight implementation of the core algorithm from the Alpa paper: https://arxiv.org/abs/2201.12023
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
        mg, module_map = alpa_autosharding_pass(mg, mesh)

    end_time = time()
    logger.info(
        f"Autosharding pass complete. Time taken: {end_time - start_time} seconds."
    )

    return mg, module_map
