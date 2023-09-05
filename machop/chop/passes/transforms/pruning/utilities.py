# Utilities (helper classes, functions, etc.) for the pruning transform

from functools import reduce

import torch
import torch.nn as nn
from pathlib import Path

from chop.tools.logger import getLogger

# Housekeeping -------------------------------------------------------------------------
logger = getLogger(__file__)
logger.propagate = False


# Model Structure Inspection & Analysis ------------------------------------------------
# Measure the sparsity of the given model
# sparsity = #zeros / #elements = 1 - #nonzeros / #elements
def measure_sparsity(model: nn.Module) -> float:
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


# Count the number of parameters; by default, all parameters are counted. The entry
# function controls what parameters make it into the list. :)
def count_parameters(model: nn.Module, nonzero_only=False, trainable_only=False) -> int:
    def count_fn(p):
        return p.count_nonzero() if nonzero_only else p.numel()

    def entry_fn(p):
        return (not trainable_only) or (trainable_only and p.requires_grad)

    return sum(count_fn(param) for param in model.parameters() if entry_fn(param))


# Buffers are similar to model parameters in that they're registered but are neither
# included in computing gradients nor optimised. Ex: batch norm's running_mean. These
# values are stored as a part of the model's state_dict.
def count_buffers(model: nn.Module) -> int:
    return sum(buffer.numel() for buffer in model.buffers())


# Returns the size of a model in MB; the memory required to store the model's parameters
# (trainable) and registered buffers (i.e. everything that's a part of the state_dict).
# NOTE: This code doesn't take mixed-precision networks into account.
def estimate_model_size(model: nn.Module, precision: int = 32) -> float:
    paramters = count_parameters(model)
    buffers = count_buffers(model)
    return (paramters + buffers) * precision / (8 * 1000**2)


# Retrieve a module nested in another by its access string. Note that this works even
# when there is a Sequential in the module. Refer to this link for more information:
# https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
def get_module_by_name(module, access_string: str):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


# Classes ------------------------------------------------------------------------------
# NOTE: Streams are just the number of input channels in our use case. :)
class StatisticsCollector:
    def __init__(self, name, streams, device, log_values=True):
        self.name = name
        self.device = device
        self.streams = streams
        self.log_values = log_values  # Values can get large, so this helps with memory

        # Internal Attributes ----------------------------------------------------------
        self.avg = torch.zeros(streams)
        self.var = torch.zeros(streams)
        self.cor = torch.zeros(streams, streams)  # Correlation matrix
        self.cov = torch.zeros(streams, streams)  # Covariance matrix
        self.cnt = 0  # Number of values processed

        # Push everything to to the appropriate device
        self.avg = self.avg.to(device)
        self.var = self.var.to(device)
        self.cor = self.cor.to(device)
        self.cov = self.cov.to(device)

    def update(self, values: torch.Tensor, id: str, save_dir: Path):
        assert values.shape[1] == self.streams, "Stream mismatch in update values!"

        # NOTE: We perform an online update of the statistics using Welford's algorithm
        # For more details, please refer to:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self.var.mul_(self.cnt)
        self.cov.mul_(self.cnt - 1)

        self.cnt += values.shape[0]
        assert self.cnt > 1, "Insufficient number of values to compute correlation!"

        delta_old = values - self.avg.view(1, -1)
        self.avg.add_(delta_old.sum(dim=0) / self.cnt)
        delta_new = values - self.avg.view(1, -1)

        self.var.add_(delta_old.mul(delta_new).sum(dim=0))
        self.var.div_(self.cnt)
        self.cov.addmm_(delta_old.t(), delta_new)

        self.cov.div_(self.cnt - 1)
        self.cor.copy_(self.cov / (self.var.outer(self.var).sqrt() * (self.cnt - 1)))
        self.cor.div_(self.cnt)

        if self.log_values:
            values_dir = save_dir / "values"
            values_dir.mkdir(parents=True, exist_ok=True)
            torch.save(values.detach().cpu(), values_dir / f"{self.name}-{id}.pt")
