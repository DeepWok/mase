import itertools
import numpy as np
import torch
import torch.nn as nn

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

logger = get_logger(__name__)

ALPA_LAYERS = {
    torch.transpose: transpose_strategy,
    torch.mm: mm_strategy,
    torch.addmm: addmm_strategy,
    torch.bmm: bmm_strategy,
    torch.baddbmm: baddmm_strategy,
}
