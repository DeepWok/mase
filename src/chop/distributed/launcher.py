import os
from functools import partial
from time import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed._tensor import (
    DeviceMesh,
    Replicate,
    Shard,
)

from chop.distributed.tensor import distribute_module, distribute_tensor

from chop.distributed.utils import rlog
from ..tools import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def distributed_timing(fn, *args, **kwargs):
    dist.barrier(async_op=True)
    start = time()
    result = fn(*args, **kwargs)
    dist.barrier(async_op=True)
    end = time()

    return result, (end - start)


def distributed_average_timing(fn, repeat, args):
    times = []
    for itr in range(repeat):
        rlog(
            logger,
            dist.get_rank(),
            f"Running teration {itr}",
            "debug",
        )
        dist.barrier(async_op=True)
        start = time()
        result = fn(*args)
        dist.barrier(async_op=True)
        end = time()
        times.append(end - start)
        rlog(
            logger,
            dist.get_rank(),
            f"Time taken: {end - start}s",
            "debug",
        )

    return result, sum(times[2:]) / len(times[2:])


def dist_model_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
    rank: int,
    tensor_sharding_map={},
) -> None:
    """
    This function gets called by torch.distributed._tensor.distribute_module on each module in the model.
    Each tensor in each module is distributed according to the sharding configuration in tensor_sharding_map.
    """
    if module in tensor_sharding_map:
        node_name = tensor_sharding_map[module]["node"]
        for parameter, sharding_config in tensor_sharding_map[module][
            "sharding"
        ].items():
            if parameter in ["data_in_0", "output", "data_out_0"]:
                continue
            if not hasattr(module, parameter):
                rlog(
                    logger,
                    rank,
                    f"Module {module} does not have parameter {parameter}",
                    level="warning",
                )
                continue

            placement = sharding_config.placements

            try:
                rlog(
                    logger,
                    rank,
                    f"Distributing parameter {parameter} of module {node_name} to {placement}",
                    level="debug",
                )
                distributed_tensor = distribute_tensor(
                    getattr(module, parameter), device_mesh, placement
                )
                setattr(module, parameter, torch.nn.Parameter(distributed_tensor))
            except Exception as e:
                rlog(
                    logger,
                    rank,
                    f"Error distributing parameter {parameter} of module {node_name} to {placement}: {e}",
                    level="error",
                )


def device_fn(
    rank, world_size, model=None, device_mesh=None, tensor_sharding_map={}, inputs=[]
):
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
    model, dist_time = distributed_timing(
        distribute_module,
        model,
        mesh,
        partial(dist_model_fn, rank=rank, tensor_sharding_map=tensor_sharding_map),
        input_fn=None,
        output_fn=None,
    )
    rlog(logger, rank, f"Module distribution done. Time taken: {dist_time} seconds.")

    # Run forward pass
    rlog(logger, rank, f"Starting forward pass.", level="info")
    inputs = [
        distribute_tensor(in_tensor, mesh, [Replicate(), Replicate()])
        for in_tensor in inputs
    ]
    _, time_taken = distributed_average_timing(
        fn=model,
        repeat=10,
        args=inputs,
    )
    rlog(logger, rank, f"Forward pass finished. Time taken: {time_taken}", level="info")

    dist.destroy_process_group()


class MaseLauncher:
    """
    MaseLauncher launches an optimized model on multiple GPUs using torch.distributed.
    """

    def __init__(self, mase_graph, world_size=None, device_mesh=None):
        """Initialize the MaseLauncher.

        Args:
            mase_graph (MaseGraph): The MaseGraph object containing the model.
            world_size (int, optional): Number of GPUs to use. Defaults to None.
            device_mesh (list, optional): List of GPUs to use. Defaults to None.
        """
        self.mg = mase_graph
        self.model = mase_graph.model
        self.world_size = world_size
        self.device_mesh = device_mesh

    def run(self, tensor_sharding_map={}, inputs=[]):
        logger.info(f"Launching model with world size {self.world_size}.")

        mp.spawn(
            partial(
                device_fn,
                model=self.model,
                device_mesh=self.device_mesh,
                tensor_sharding_map=tensor_sharding_map,
                inputs=inputs,
            ),
            args=(self.world_size,),
            nprocs=self.world_size,
            join=True,
        )
