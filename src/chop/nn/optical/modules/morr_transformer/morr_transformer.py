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

from ...utils import MORRConfig_20um_MQ
from ...utils import mrr_roundtrip_phase_to_tr_func, mrr_roundtrip_phase_to_tr_fused
from ...utils import toeplitz
from ...utils import morr_uniform_
from ...utils import input_quantize_fn, weight_quantize_fn
from ..base_layer import ONNBaseLayer
from ..morr_custom_linear import AllPassMORRLinear
from ..morr_linear import AllPassMORRCirculantLinear
from .morr_matmul import AllPassMORRCirculantMatMuls

from transformers import BertModel, BertForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
    Conv1D,
)

logger = logging.getLogger(__name__)

__all__ = [""]



class MORRMHA(nn.Module):
    def __init__(self, embed_dim, heads):
        super(MORRMHA, self).__init__()
        assert embed_dim % heads == 0
        self.n_heads = heads
        self.Wq = AllPassMORRCirculantLinear(embed_dim, embed_dim)
        self.Wk = AllPassMORRCirculantLinear(embed_dim, embed_dim)
        self.Wv = AllPassMORRCirculantLinear(embed_dim, embed_dim)
        self.qmm1 = AllPassMORRCirculantMatMuls()
        self.dropout_wq = nn.Dropout(0.1)
        self.dropout_wk = nn.Dropout(0.1)
        self.dropout_wv = nn.Dropout(0.1)
        self.qmm2 = AllPassMORRCirculantMatMuls()
        self.Wout = AllPassMORRCirculantLinear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask):
        b = x.size(0)
        n = x.size(1)
        h = self.n_heads
        d = x.size(2)

        def arrange_heads(acts):
            # incoming shape of b, n, d, want b, h, n, d/h
            return acts.view(b, n, h, -1).transpose(1, 2)

        q = arrange_heads(self.dropout_wq(self.Wq(x)))
        k = arrange_heads(self.dropout_wk(self.Wk(x)))
        v = arrange_heads(self.dropout_wv(self.Wv(x)))

        attn = self.qmm1(q, k.transpose(2, 3)) # yields b, h, n, n
        masked = attn.masked_fill(mask, float("-inf"))
        softmax_attn = self.dropout1(F.softmax(masked / math.sqrt(d // h), dim=3))
        out = self.qmm2(softmax_attn, v) # b, h, n, d/h

        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.dropout2(out)
        out = self.Wout(out)
        return out


class MORRFF(nn.Module):
    def __init__(self, embed_dim, expansion_dim):
        super(MORRFF, self).__init__()
        self.first_drop = nn.Dropout(0.1)
        self.layer1 = AllPassMORRCirculantLinear(embed_dim, expansion_dim, use_noise=True)
        self.act = nn.ReLU6(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.layer2 = AllPassMORRCirculantLinear(expansion_dim, embed_dim, use_noise=True)

    def forward(self, x):
        out = self.first_drop(x)
        out = self.layer1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out

class MORRDecoderLayer(nn.Module):
    def __init__(self, features, heads):
        super(MORRDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(features)
        self.attn = MORRMHA(features, heads)
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(features)
        self.ff = MORRFF(features, features * 4)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x, attn_mask):
        # no need for key mask for gpt; autoregressive masking already prevents 'real' tokens from attending to padding tokens to the right
        identity = x
        out = self.norm1(x)
        out = self.attn(out, attn_mask)
        out = self.drop1(out)
        out = out + identity
        identity = out
        out = self.norm2(out)
        out = self.ff(out)
        out = self.drop2(out)
        out = out + identity
        return out


class MORRSdpa(nn.Module):
    def __init__(self, attn_head_size, num_heads, seq_length, dropout_p, use_morr = False, morr_config = None):
        super(MORRSdpa, self).__init__()
        self.attn_head_size = attn_head_size
        self.num_heads = num_heads
        self.use_morr = use_morr
        self.qmm1 = AllPassMORRCirculantMatMuls(
            in_features=attn_head_size, # Dh 
            out_features=seq_length, # N
            config = morr_config
        )
        self.qmm1.disable_trainable_morr_scale()
        self.qmm1.disable_trainable_morr_bias()
        
        self.qmm2 = AllPassMORRCirculantMatMuls(
            in_features=seq_length, # D 
            out_features=attn_head_size, # N
            config = morr_config
        )
        self.qmm2.disable_trainable_morr_scale()
        self.qmm2.disable_trainable_morr_bias()
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, query, key, value, attn_mask):
        attn_head_size = self.attn_head_size

        if self.use_morr:
            attn_scores = self.qmm1(query, key.transpose(2, 3)) # yields b, h, n, n
        else:
            attn_scores = torch.matmul(query, key.transpose(2, 3))
            
        attn_scores = attn_scores / math.sqrt(attn_head_size)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        if self.use_morr:
            out = self.qmm2(attn_probs, value) # [B, H, N, N] * [B, H, N, Dh] -> [b, h, n, Dh]
        else:
            out = torch.matmul(attn_probs, value)

        return out