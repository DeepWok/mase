
import torch
import torch.nn as nn

from torch.distributed._tensor import (
    DeviceMesh,
)

from torch.distributed._tensor.api import Redistribute

from chop.distributed.utils import placement_from_sharding_config, rlog
from chop.tools import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")

class ReshardingWrapper(nn.Module):
    def __init__(self, device_mesh, module, resharding_config):
        super().__init__()
        self.module = module
        self.resharding_config = resharding_config
        self.device_mesh = device_mesh

    def forward(self, x):
        rank = torch.distributed.get_rank()
        device_mesh = DeviceMesh("cuda", self.device_mesh)

        required_placement = placement_from_sharding_config(self.resharding_config["input"])
        if (x.placements != required_placement):
            rlog(logger, rank, f"For module {self.module}, resharding tensor x from {x.placements} to {required_placement}", level="debug")
            x = Redistribute.apply(x, device_mesh, required_placement)

        return self.module(x)

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
        module = getattr(mg.model, node.target, None)
        if module is not None:
            resharding_config = module_map[module]
            setattr(mg.model, node.target, ReshardingWrapper(device_mesh, module, resharding_config))
    
    return mg, {}