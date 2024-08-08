from time import time


import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh

from chop.tools import get_logger
from chop.distributed.tensor import distribute_tensor

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def rlog(logger, rank, msg, level="info"):
    """
    Only log on rank 0 to avoid repeated messages.
    """
    log_fn = getattr(logger, level, logger.info)
    if rank == 0:
        log_fn(msg)


def distributed_timing(fn, *args, **kwargs):
    dist.barrier(async_op=True)
    start = time()
    result = fn(*args, **kwargs)
    dist.barrier(async_op=True)
    end = time()

    return result, (end - start)


def distributed_average_timing(
    fn,
    args,
    repeat=10,
    warmup_iters=2,
):
    times = []
    for itr in range(repeat):
        rlog(
            logger,
            dist.get_rank(),
            f"Running teration {itr}",
            "info",
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
            "info",
        )

    return result, sum(times[warmup_iters:]) / len(times[warmup_iters:])


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

    Args:
        name (str): _description_
        module (nn.Module): _description_
        device_mesh (DeviceMesh): _description_
        rank (int): _description_
        tensor_sharding_map (dict, optional): _description_. Defaults to {}.
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
                setattr(
                    module,
                    parameter,
                    torch.nn.Parameter(distributed_tensor),
                )
            except Exception as e:
                rlog(
                    logger,
                    rank,
                    f"Error distributing parameter {parameter} of module {node_name} to {placement}: {e}",
                    level="error",
                )
                raise e
