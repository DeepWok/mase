# Utilities (helper classes, functions, etc.) for the pruning transform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from chop.tools.logger import getLogger

# Housekeeping -------------------------------------------------------------------------
logger = getLogger(__file__)
logger.propagate = False


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
