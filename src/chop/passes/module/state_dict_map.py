import inspect
import re
import os
from copy import deepcopy
from typing import Tuple

from chop.nn.quantizers.SNN.LSQ import LSQInteger
from chop.nn.quantized.modules.roberta.attention import RobertaSelfAttentionLSQInteger
from chop.nn.snn.modules.linear import LinearUnfoldBias
from chop.nn.snn.modules.roberta.attention import RobertaSelfAttentionZIPTF
from chop.nn.mla.modules.model import MLA

from chop.nn.snn.modules.neuron.st_bifnode import ST_BIFNode
import torch
from pathlib import Path
from functools import reduce
from transformers import PreTrainedModel, TFPreTrainedModel
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSdpaSelfAttention

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

def bert_attn_convert(sdpa_attn: BertSdpaSelfAttention, mla_attn: MLA) -> MLA:
    """
    Convert (copy) the core parameters from a BertSdpaSelfAttention module into an MLA module.
    NOTE: This is a skeleton showing the mapping approach. Actual shape matching and LoRA logic
          may require extra steps depending on your configuration.
    """

    # ---------------------------------------
    # 1. Handle the "query" -> "wq" transfer
    # ---------------------------------------
    # If MLA is configured for no LoRA in Q,
    # simply copy from Bert's query weights/bias:
    if mla_attn.q_lora_rank == 0:
        # Typically: sdpa_attn.query.weight has shape [hidden_size, hidden_size]
        # mla_attn.wq has shape [hidden_size, n_heads * qk_head_dim]
        # Make sure these match or reshape accordingly.
        mla_attn.wq.weight.data.copy_(sdpa_attn.query.weight.data)
        mla_attn.wq.bias.data.copy_(sdpa_attn.query.bias.data)
    else:
        # Otherwise, if you are using LoRA for Q:
        #   wq_a: [hidden_size, q_lora_rank]
        #   wq_b: [q_lora_rank, n_heads * qk_head_dim]
        #   q_norm: RMSNorm(q_lora_rank)
        #
        # You cannot directly copy [hidden_size -> hidden_size] into these.
        # Instead, you might factor or approximate your original query weights
        # or do some custom initialization. Here is a placeholder:
        with torch.no_grad():
            # For example, just fill them with a truncated or random approach
            # or do your own decomposition from the original matrix:
            # wq_a in [dim, q_lora_rank]
            mla_attn.wq_a.weight.data.zero_()  # placeholder
            mla_attn.wq_a.bias.data.zero_()
            
            # wq_b in [q_lora_rank, n_heads*qk_head_dim]
            mla_attn.wq_b.weight.data.zero_()
            mla_attn.wq_b.bias.data.zero_()
            
            # If your design is to factorize or approximate,
            # you'll do that decomposition here instead of zeros.

    # ---------------------------------------
    # 2. Handle the "key" + "value" -> "wkv_a/b" transfer
    # ---------------------------------------
    # The new MLA code merges key/value into a single pair of linear layers:
    #   wkv_a: [dim, kv_lora_rank + qk_rope_head_dim]
    #   wkv_b: [kv_lora_rank, n_heads*(qk_nope_head_dim + v_head_dim)]
    #
    # In standard BERT, key and value each have shape [hidden_size, hidden_size].
    #
    # Since you are merging them, you need to slice or reshape accordingly.
    # Example (no LoRA for K/V):
    if mla_attn.kv_lora_rank == 0:
        # The rope part (k_pe) is appended in the forward pass. 
        # `wkv_a` can store partial weights (for the key) plus space for rope dimension.
        # `wkv_b` merges the transform for both key and value.
        # 
        # Typically you'd do something like:
        with torch.no_grad():
            # For demonstration, we just fill them from the BERT's key and value
            # but you must ensure your shapes match your qk_rope_head_dim, etc.
            # The simplest approach is if qk_rope_head_dim=0 and you treat wkv_a/b
            # as standard linear layers for K/V.
            
            # Suppose you do not use rope at all (qk_rope_head_dim=0) => wkv_a is [dim, kv_lora_rank=0]
            # That effectively kills wkv_a =>  it might not even exist or be a zero shape.
            # Then wkv_b => [0, ???], also a mismatch. 
            # In a naive scenario, you'd have a single linear [dim, 2*hidden_size] to store (K||V).
            # 
            # This example is purely illustrative:
            mla_attn.wkv_a.weight.data.zero_()
            mla_attn.wkv_a.bias.data.zero_()
            
            # If you want to store key + value in wkv_b, you might do a split:
            # old_key = sdpa_attn.key.weight  -> shape [hidden_size, hidden_size]
            # old_val = sdpa_attn.value.weight -> shape [hidden_size, hidden_size]
            # new_kv  -> shape [hidden_size, 2*hidden_size] if you wanted them concatenated
            # Then reshape that to match your final wkv_b shape in MLA.
            # For demonstration only:
            combined_kv = torch.cat([sdpa_attn.key.weight.data, sdpa_attn.value.weight.data], dim=0)
            # This combined_kv is now shape [2*hidden_size, hidden_size].
            # You'd likely want to reshape or transpose to match wkv_b's shape exactly.
            # So, again, do that carefully here:
            b_weight = mla_attn.wkv_b.weight.data
            # e.g. if b_weight is [kv_lora_rank, n_heads*(qk_nope_head_dim + v_head_dim)],
            # you need the same total number of elements. 
            # This is just a placeholder:
            flattened = combined_kv.flatten()[: b_weight.numel()]
            b_weight.copy_(flattened.view_as(b_weight))
            mla_attn.wkv_b.bias.data.zero_()
    else:
        # LoRA scenario for K/V
        # wkv_a: [dim, kv_lora_rank + qk_rope_head_dim]
        # wkv_b: [kv_lora_rank, n_heads*(qk_nope_head_dim + v_head_dim)]
        # 
        # You again need some form of decomposition or custom approach.
        # Here, just placeholder:
        with torch.no_grad():
            mla_attn.wkv_a.weight.data.zero_()
            mla_attn.wkv_a.bias.data.zero_()
            mla_attn.kv_norm.weight.data.fill_(1.0)  # RMSNorm scale
            mla_attn.kv_norm.bias.data.zero_()
            mla_attn.wkv_b.weight.data.zero_()
            mla_attn.wkv_b.bias.data.zero_()

    # ---------------------------------------
    # 3. Handle the final output projection
    # ---------------------------------------
    # In standard BERT, the "output" projection is typically `self.dense` in
    # `BertSelfOutput` or so, but you might want to match that with MLA's `wo`.
    # If you want a direct copy from the old "dense" weight, do:
    # (Your BertSdpaSelfAttention might not define .dense directly; 
    #  it might be in the parent module.)
    if hasattr(sdpa_attn, "dense"):
        mla_attn.wo.weight.data.copy_(sdpa_attn.dense.weight.data)
        mla_attn.wo.bias.data.copy_(sdpa_attn.dense.bias.data)
    else:
        # If your original code has "output projection" somewhere else, 
        # you can manually fetch it. Or do a placeholder:
        mla_attn.wo.weight.data.zero_()
        mla_attn.wo.bias.data.zero_()

    return mla_attn


SPECIAL_CONVERT_PATTERNS = {
    (RobertaSelfAttentionLSQInteger, RobertaSelfAttentionZIPTF): attn_convert,
    (LSQInteger, ST_BIFNode): lsqinteger_to_st_bif,
    (BertSelfAttention, BertSdpaSelfAttention): bert_attn_convert,
}
