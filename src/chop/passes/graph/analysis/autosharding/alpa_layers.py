import itertools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from chop.tools import get_logger
from chop.models.patched.bert.modeling_bert import BertSelfAttention

from .alpa_cost_modelling import get_communication_cost

from .ops.matrix_ops import (
    transpose_strategy,
    mm_strategy,
    addmm_strategy,
    bmm_strategy,
    baddmm_strategy,
)

from .ops.view_ops import get_reshape_strategy
from .ops.pointwise_ops import linear_pointwise_strategy

logger = get_logger(__name__)

ALPA_FUNCTIONS = {
    torch.transpose: transpose_strategy,
    torch.mm: mm_strategy,
    torch.addmm: addmm_strategy,
    torch.bmm: bmm_strategy,
    torch.baddbmm: baddmm_strategy,
    torch.add: linear_pointwise_strategy,
    operator.add: linear_pointwise_strategy
}

ALPA_METHODS = {
    "view": get_reshape_strategy(torch.Tensor.view),
    "expand": get_reshape_strategy(torch.Tensor.expand),
    "permute": get_reshape_strategy(torch.permute),
}
