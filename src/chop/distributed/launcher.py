import os
from functools import partial
from time import time

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

from chop.distributed.utils import rlog
from ..tools import get_logger
from .utils import placement_from_sharding_config

logger = get_logger(__name__)
logger.setLevel("INFO")

def distributed_timing(fn, *args, **kwargs):
    dist.barrier()
    start = time()
    result = fn(*args, **kwargs)
    dist.barrier()
    end = time()
    return result, (end - start)

def dist_model_fn(
    name: str, module: nn.Module, device_mesh: DeviceMesh, rank: int, module_map={}
) -> None:
    """
    This function gets called by torch.distributed._tensor.distribute_module on each module in the model.
    Each tensor in each module is distributed according to the sharding configuration in module_map.
    """
    if module in module_map:
        node_name = module_map[module]["node"]
        for parameter, sharding_config in module_map[module]["sharding"].items():
            if parameter in ["data_in_0", "output", "data_out_0"]:
                continue
            if not hasattr(module, parameter):
                rlog(logger, rank, f"Module {module} does not have parameter {parameter}", level="warning")
                continue
            
            placement = placement_from_sharding_config(sharding_config)

            try:
                rlog(logger, rank, f"Distributing parameter {parameter} of module {node_name} to {placement}", level="debug")
                distributed_tensor = distribute_tensor(getattr(module, parameter), device_mesh, placement)
                setattr(module, parameter, torch.nn.Parameter(distributed_tensor))
            except Exception as e:
                rlog(logger, rank, f"Error distributing parameter {parameter} of module {node_name} to {placement}: {e}", level="error")


def device_fn(rank, world_size, model=None, device_mesh=None, module_map={}, inputs=[]):
    """
    This function gets called on each GPU device to set up the distributed environment and distribute the model,
    following the SPMD model.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)

    # Initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    # Distribute model parameters according to sharding configuration
    mesh = DeviceMesh("cuda", mesh=device_mesh)
    rlog(logger, rank, f"Distributing module parameters...", level="info")
    model, dist_time = distributed_timing(distribute_module, model, mesh, partial(dist_model_fn, rank=rank, module_map=module_map), input_fn=None, output_fn=None)
    rlog(logger, rank, f"Module distribution done. Time taken: {dist_time} seconds.")

    # Run forward pass
    rlog(logger, rank, f"Starting forward pass.", level="info")
    inputs = [distribute_tensor(in_tensor, mesh, [Replicate(), Replicate()]) for in_tensor in inputs]
    out, time_taken = distributed_timing(model, *inputs)
    rlog(logger, rank, f"Forward pass finished. Time taken: {time_taken}", level="info")

    dist.destroy_process_group()

class MaseLauncher():
    def __init__(self, mase_graph, world_size = None, device_mesh=None):
        self.mg = mase_graph
        self.model = mase_graph.model
        self.world_size = world_size
        self.device_mesh = device_mesh

    def run(self, module_map = {}, inputs=[]):
        logger.info(f"Launching model with world size {self.world_size}.")
        mp.spawn(partial(device_fn, model=self.model, device_mesh=self.device_mesh, module_map=module_map, inputs=inputs), args=(self.world_size,), nprocs=self.world_size, join=True)