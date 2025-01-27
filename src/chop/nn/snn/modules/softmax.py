import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import math

import chop.nn.snn.base as base


class SoftmaxZIPTF(nn.Softmax, base.StepModule):
    """
    Stateful Softmax function
    Copied from SpikeZIP-TF
    https://arxiv.org/pdf/2406.03470
    """

    def __init__(self, dim=-1, step_mode="s") -> None:
        super().__init__(dim=dim)
        self.X = 0.0
        self.Y_pre = 0.0
        self.step_mode = step_mode

    def reset(self):
        self.X = 0.0
        self.Y_pre = 0.0

    def forward(self, input):
        if self.step_mode == "s":
            self.X = input + self.X
            Y = super().forward(self.X)
            Y_pre = deepcopy(self.Y_pre)
            self.Y_pre = Y
            return Y - Y_pre

        elif self.step_mode == "m":
            T = input.shape[0]
            y_seq = []
            for t in range(T):
                self.X = input[t] + self.X
                Y = super().forward(self.X)
                Y_pre = deepcopy(self.Y_pre)
                self.Y_pre = Y
                y_seq.append(Y - Y_pre)
            return torch.stack(y_seq, dim=0)
