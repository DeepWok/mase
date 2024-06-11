import os
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed._tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)

from chop.tools import deepsetattr, get_logger

from .utils import placement_from_sharding_config

logger = get_logger(__name__)
logger.setLevel("DEBUG")

def rlog(rank, msg):
    """
    Only log on rank 0 to avoid repeated messages.
    """
    if rank == 0:
        logger.info(msg)

def dist_model_fn(
    name: str, module: nn.Module, device_mesh: DeviceMesh, rank: int, module_map={}
) -> None:
    """
    This function gets called by torch.distributed._tensor.distribute_module on each module in the model.
    Each tensor in each module is distributed according to the sharding configuration in module_map.
    """
    rlog(rank, f"Processing module {module}")
    if module in module_map:
        for parameter, sharding_config in module_map[module].items():
            rlog(rank, f"    Parameter: {parameter} has sharding config {sharding_config}")
            if not hasattr(module, parameter):
                rlog(rank, f"    Module does not have parameter {parameter}")
                continue
            placement = placement_from_sharding_config(sharding_config)
            rlog(rank, f"    Placement: {placement}")
            deepsetattr(module, parameter, torch.nn.Parameter(distribute_tensor(getattr(module, parameter), device_mesh, placement)))


def device_fn(rank, world_size, model=None, device_mesh=None, module_map={}, inputs=[]):
    """
    This function gets called on each GPU device to set up the distributed environment and distribute the model,
    following the SPMD model.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    mesh = DeviceMesh("cuda", mesh=device_mesh)
    model = distribute_module(
        model, mesh, partial(dist_model_fn, rank=rank, module_map=module_map), input_fn=None, output_fn=None
    )

    # TO DO: read from module_map with keys matching forward function signature
    inputs = [distribute_tensor(in_tensor, mesh, [Replicate(), Replicate()]) for in_tensor in inputs]
    out = model(*inputs)

    # TO DO: how to return output?

    rlog(rank, f"Module distribution done.")

    dist.destroy_process_group()

class MaseLauncher():
    def __init__(self, mase_graph, world_size = None, device_mesh=None):
        self.mg = mase_graph
        self.model = mase_graph.model
        self.world_size = world_size
        self.device_mesh = device_mesh

    def run(self, module_map = {}, inputs=[]):
        mp.spawn(partial(device_fn, model=self.model, device_mesh=self.device_mesh, module_map=module_map, inputs=inputs), args=(self.world_size,), nprocs=self.world_size, join=True)