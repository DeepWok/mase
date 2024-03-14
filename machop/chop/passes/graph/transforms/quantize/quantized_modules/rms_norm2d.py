import logging

import torch
import torch.nn as nn
from torch import Tensor

from ..quantizers.integer import integer_floor_quantizer


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class _RMSNormBase(nn.Module):
    def __init__(
        self,
        normalized_shape,
    ) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        pass
