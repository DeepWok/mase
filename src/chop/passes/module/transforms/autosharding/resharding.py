
import functools

import torch
import torch.nn as nn

from torch.distributed._tensor import (
    DeviceMesh,
)

from torch.distributed._tensor.api import Redistribute

from chop.distributed.utils import placement_from_sharding_config
from chop.tools import get_logger

logger = get_logger(__name__)
logger.setLevel("INFO")

def rlog(logger, rank, msg, level="info"):
    """
    Only log on rank 0 to avoid repeated messages.
    """
    log_fn = getattr(logger, level, logger.info)
    if (rank == 0):
        log_fn(f"[RANK: {rank}]: {msg}")

def deepsetattr(obj, attr, value):
    """Recurses through an attribute chain to set the ultimate value."""
    attrs = attr.split(".")
    if len(attrs) > 1:
        deepsetattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]), value)
    else:
        setattr(obj, attr, value)

def deepgetattr(obj, attr, default=None):
    """Recurses through an attribute chain to get the ultimate value."""
    try:
        return functools.reduce(getattr, attr.split("."), obj)
    except AttributeError:
        return default

class ReshardingWrapper(nn.Module):
    def __init__(self, device_mesh, module, resharding_config):
        super().__init__()
        self.module = module
        self.resharding_config = resharding_config["sharding"]
        self.node = resharding_config["node"]
        self.device_mesh = device_mesh

    def forward(self, x):
        rank = torch.distributed.get_rank()
        device_mesh = DeviceMesh("cuda", self.device_mesh)

        required_placement = placement_from_sharding_config(self.resharding_config["data_in_0"])
        if (x.placements != required_placement):
            rlog(logger, rank, f"For module {self.node}, resharding tensor x from {x.placements} to {required_placement}", level="debug")
            x = Redistribute.apply(x, device_mesh, required_placement)

        out = self.module(x)
        
        return out

def resharding_transform_pass(mg, pass_args={}):
    """
    This pass inserts a wrapper around each module in the graph to handle resharding
    activation tensors when the output of the previous module has a different sharding
    profile to the one assigned to the current module.
    """

    module_map = pass_args.get("module_map", None)
    device_mesh = pass_args.get("device_mesh", None)
    if module_map is None or device_mesh is None:
        raise ValueError("module_map and device_mesh are required for resharding_transform_pass")

    for node in mg.fx_graph.nodes:
        if node.op != "call_module":
            continue
        module = deepgetattr(mg.model, node.target, None)
        if module is not None:
            resharding_config = module_map[module]
            logger.debug(f"Inserting resharding wrapper around node: {node}")
            deepsetattr(mg.model, node.target, ReshardingWrapper(device_mesh, module, resharding_config))

    mg.model.recompile()
    
    return mg, {}