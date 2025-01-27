import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import math


# TODO: this need to be change to neurons model later if we want to support training
class ST_BIFNode(nn.Module):
    def __init__(self, q_threshold=torch.tensor(1), level=32, sym=False):
        super(ST_BIFNode, self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = (
            torch.tensor(q_threshold)
            if not torch.is_tensor(q_threshold)
            else q_threshold
        )
        self.is_work = False
        self.cur_output = 0.0
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.pos_max = torch.tensor(level // 2 - 1)
            self.neg_min = torch.tensor(-level // 2)
        else:
            self.pos_max = torch.tensor(level - 1)
            self.neg_min = torch.tensor(0)

        self.eps = 0

    def __repr__(self):
        return f"ST_BIFNode(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"

    def reset(self):
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        # I believe this is some shot of early stopping machanism
        self.is_work = False
        self.spike_position = None
        self.neg_spike_position = None

    def forward(self, input):
        x = input / self.q_threshold
        if (
            (not torch.is_tensor(x))
            and x == 0.0
            and (not torch.is_tensor(self.cur_output))
            and self.cur_output == 0.0
        ):
            self.is_work = False
            return x

        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape, dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape).to(x.device)
            self.q = torch.zeros(x.shape).to(x.device) + 0.5

        self.is_work = True

        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0) & (self.acc_q < self.pos_max)
        neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        if (x == 0).all() and (self.cur_output == 0).all():
            self.is_work = False

        return self.cur_output * self.q_threshold
