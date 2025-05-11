from typing import Optional
import logging

import numpy as np
import math
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device
import pytorch_lightning as pl
import torchmetrics
import transformers
from transformers import GPT2TokenizerFast
from packaging import version
from typing import List, Optional, Tuple, Union

from ...utils import MORRConfig_20um_MQ
from ...utils import mrr_roundtrip_phase_to_tr_func, mrr_roundtrip_phase_to_tr_fused
from ...utils import toeplitz
from ...utils import morr_uniform_
from ...utils import input_quantize_fn, weight_quantize_fn
from ..base_layer import ONNBaseLayer
from ..morr_custom_linear import AllPassMORRLinear
from ..morr_linear import AllPassMORRCirculantLinear
from .morr_matmul import AllPassMORRCirculantMatMuls
from .morr_transformer import MORRSdpa

from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.utils import (
    get_torch_version,
)

class BertMORRSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None, morr_config=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")
        # define MORR object to perform SDPA
        self.morr_spda = None
        self.morr_config = morr_config
    
    # Adapted from BertSelfAttention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
            # logger.warning_once(
            #     "BertSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
            #     "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
            #     "the manual attention implementation, but specifying the manual implementation will be required from "
            #     "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
            #     '`attn_implementation="eager"` when loading the model.'
            # )
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(current_states))
            value_layer = self.transpose_for_scores(self.value(current_states))
            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = (
            True if self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1 else False
        )

        self.morr_spda = MORRSdpa(
            self.attention_head_size, # Dh
            self.num_attention_heads, # H
            hidden_states.shape[1], # N
            dropout_p=self.dropout_prob,
            use_morr=True,
            morr_config=self.morr_config,
        )
        attn_output = self.morr_spda(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs