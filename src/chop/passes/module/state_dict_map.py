import inspect
import re
import os
from copy import deepcopy
from typing import Tuple

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantized.modules.roberta.attention import RobertaSelfAttentionLSQInteger
from chop.nn.snn.modules.linear import LinearUnfoldBias
from chop.nn.snn.modules.roberta.attention import RobertaSelfAttentionZIPTF

from chop.nn.snn.modules.neuron.st_bifnode import ST_BIFNode
import torch
from pathlib import Path
from functools import reduce
from transformers import PreTrainedModel, TFPreTrainedModel


def match_a_pattern(name: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return pattern
    return None


def check_is_huggingface_model(model):
    return isinstance(model, (PreTrainedModel, TFPreTrainedModel))


def attn_convert(
    QAttn: RobertaSelfAttentionLSQInteger, SAttn: RobertaSelfAttentionZIPTF
) -> RobertaSelfAttentionZIPTF:
    # NOTE: level and neuron_type are configure during the initialization of the module through the config args
    level = SAttn.level
    neuron_type = SAttn.neuron_type

    SAttn.query = LinearUnfoldBias(
        in_features=QAttn.query.in_features,
        out_features=QAttn.query.out_features,
        bias=QAttn.query.bias is not None,
        neuron_type="ST-BIF",
        level=level,
    )
    SAttn.query.weight.data = QAttn.query.weight.data
    SAttn.query.bias.data = QAttn.query.bias.data

    SAttn.key = LinearUnfoldBias(
        in_features=QAttn.key.in_features,
        out_features=QAttn.key.out_features,
        bias=QAttn.key.bias is not None,
        neuron_type="ST-BIF",
        level=level,
    )
    SAttn.key.weight.data = QAttn.key.weight.data
    SAttn.key.bias.data = QAttn.key.bias.data

    SAttn.value = LinearUnfoldBias(
        in_features=QAttn.value.in_features,
        out_features=QAttn.value.out_features,
        bias=QAttn.value.bias is not None,
        neuron_type="ST-BIF",
        level=level,
    )
    SAttn.value.weight.data = QAttn.value.weight.data
    SAttn.value.bias.data = QAttn.value.bias.data

    SAttn.query_IF.neuron_type = neuron_type
    SAttn.query_IF.level = level
    SAttn.query_IF.q_threshold = QAttn.query_quan.s.data
    SAttn.query_IF.pos_max = QAttn.query_quan.pos_max
    SAttn.query_IF.neg_min = QAttn.query_quan.neg_min
    SAttn.query_IF.is_init = False

    SAttn.key_IF.neuron_type = neuron_type
    SAttn.key_IF.level = level
    SAttn.key_IF.q_threshold = QAttn.key_quan.s.data
    SAttn.key_IF.pos_max = QAttn.key_quan.pos_max
    SAttn.key_IF.neg_min = QAttn.key_quan.neg_min
    SAttn.key_IF.is_init = False

    SAttn.value_IF.neuron_type = neuron_type
    SAttn.value_IF.level = level
    SAttn.value_IF.q_threshold = QAttn.value_quan.s.data
    SAttn.value_IF.pos_max = QAttn.value_quan.pos_max
    SAttn.value_IF.neg_min = QAttn.value_quan.neg_min
    SAttn.value_IF.is_init = False

    SAttn.attn_IF.neuron_type = neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold = QAttn.attn_quan.s.data
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min
    SAttn.attn_IF.is_init = False

    SAttn.after_attn_IF.neuron_type = neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    SAttn.after_attn_IF.is_init = False

    return SAttn


def lsqinteger_to_st_bif(LSQ: LSQInteger, ST_BIF: ST_BIFNode) -> ST_BIFNode:

    ST_BIF.q_threshold = LSQ.s.data
    ST_BIF.sym = LSQ.sym
    ST_BIF.level = LSQ.level
    ST_BIF.pos_max = LSQ.pos_max
    ST_BIF.neg_min = LSQ.neg_min
    ST_BIF.is_init = False

    return ST_BIF


SPECIAL_CONVERT_PATTERNS = {
    (RobertaSelfAttentionLSQInteger, RobertaSelfAttentionZIPTF): attn_convert,
    (LSQInteger, ST_BIFNode): lsqinteger_to_st_bif,
}
