from typing import Tuple

import torch
import torch.fx as fx
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed._tensor._redistribute import redistribute_local_tensor

from torch.distributed._tensor.placement_types import Placement

from chop.distributed.tensor import DTensor
from chop.tools import get_logger
from chop.distributed.utils import rlog

logger = get_logger(__name__)
logger.setLevel("DEBUG")


@fx.wrap
def dtensor_arange(
    start: int,
    end: int,
    step: int = 1,
    out: torch.Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False,
    device_mesh: DeviceMesh = None,
):
    """Returns a fully replicated DTensor with behaviour akin to `torch.arange`.

    Args:
        start (int): _description_
        end (int): _description_
        step (int, optional): _description_. Defaults to 1.
        out (torch.Tensor, optional): _description_. Defaults to None.
        dtype (torch.dtype, optional): _description_. Defaults to None.
        layout (torch.layout, optional): _description_. Defaults to torch.strided.
        device (torch.device, optional): _description_. Defaults to None.
        requires_grad (bool, optional): _description_. Defaults to False.
    """
    return DTensor.from_local(
        torch.arange(
            start,
            end,
            step,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
        ),
        device_mesh=device_mesh,
    )


@fx.wrap
def redistribute_dtensor(
    input: DTensor,
    placements: Tuple[Placement, ...],
    async_op: bool = False,
):
    """
    Redistribute a DTensor to a new set of placements.

    Args:
        input (DTensor): The input DTensor to redistribute.
        placements (Tuple[Placement, ...]): The new placements for the output DTensor.
        async_op (bool, optional): Whether to perform the redistribution asynchronously. Defaults to False.

    Returns:
        DTensor: The redistributed DTensor.
    """

    # If we are not in a distributed setting, we can skip redistribution.
    try:
        rank = torch.distributed.get_rank()
    except:
        rank = 0

    if not isinstance(input, DTensor):
        rlog(
            logger,
            rank,
            f"Skipping redistribution because received {type(input)} instead of DTensor",
            level="warning",
        )
        return input

    current_spec = input._spec

    rlog(
        logger,
        rank,
        f"Redistributing tensor from {current_spec.placements} to {placements}",
        level="info",
    )

    if current_spec.placements != placements:
        target_spec = DTensorSpec(
            input._spec.mesh,
            placements,
            tensor_meta=input._spec.tensor_meta,
        )

        local_tensor = input._local_tensor
        output = redistribute_local_tensor(
            local_tensor,
            current_spec,
            target_spec,
            async_op=async_op,
        )
    else:
        # use the same local tensor if placements are the same.
        output = input._local_tensor
        target_spec = current_spec

    return DTensor(
        output,
        target_spec,
        requires_grad=input.requires_grad,
    )
