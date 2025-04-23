import copy
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union
# from clip.model import QuickGELU
# from torch.autograd import Variable
# from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent  # this gets the current file's directory
PREMODEL_DIR = BASE_DIR / 'premodels'  # relative to your script location
INV_PATH = PREMODEL_DIR / 'distilled_inv_64.pth'
EXP_PATH = PREMODEL_DIR / 'distilled_exp_32.pth'
SQRTINV_PATH = PREMODEL_DIR / 'distilled_sqrtinv_8.pth'
GELU_PATH = PREMODEL_DIR / 'distilled_gelu_64.pth'

class StraightThrough(nn.Module):
    
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class SpikeLinear_ReLU(nn.Module):
    def __init__(self, module: nn.Module, T:int = None, **kwargs):
        super(SpikeLinear_ReLU, self).__init__()
        # module = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=DEVICE)
        self.T = T if T is not None else kwargs.get('T', 32)
        self.t = 0
        self.threshold = None
        self.mem_pot = 0
        self.mem_pot_init = 0
        self.use_spike = False
        self.relu = StraightThrough()
        self.fwd_func = F.linear
        self.weight = module.weight
        self.org_weight = copy.deepcopy(module.weight.data)
        self.bias = module.bias
        self.memory_spike = 0
        self.bipolar_with_memory = False
        self.burst_T = 0
        self.belong_to_ln = False
        self.belong_to_x2x = False
        self.belong_to_x2x_pos = False
        if module.bias is not None:
            self.bias = module.bias
            self.org_bias = copy.deepcopy(module.bias.data)
        else:
            self.bias = None
            self.org_bias = None

    def forward(self, input: torch.Tensor):
        if self.use_spike and not isinstance(self.relu, StraightThrough):

            x = self.fwd_func(input, self.weight, self.bias)
            
            # set multi-scale threshold to reduce quantization error
            Vth_scale = torch.tensor([1.0]).to(input.device)
            
            self.mem_pot = self.mem_pot + x

            spike = 0
            for i in range(len(Vth_scale)):
                Vth_lower = self.threshold * Vth_scale[i]
                Vth_upper = self.threshold * Vth_scale[i+1] if i != len(Vth_scale)-1 else 1.0e+5
                if not self.bipolar_with_memory:
                    spike += torch.logical_and(self.mem_pot >= Vth_lower, self.mem_pot < Vth_upper).float() * Vth_lower
                else:
                    spike += torch.logical_and(self.mem_pot >= Vth_lower, self.mem_pot < Vth_upper).float() * Vth_lower
                    spike += torch.logical_and(torch.logical_and(self.mem_pot <= -Vth_lower, self.mem_pot > -Vth_upper), self.memory_spike > torch.zeros_like(self.mem_pot)).float() * (-Vth_lower)
                
            self.mem_pot -= spike
            self.memory_spike += spike

            # use burst spikes with half-scale threshold to reduce both quantization error and residual potential
            Vth_burst = self.threshold * 0.5
            if self.burst_T:
                if self.bipolar_with_memory:
                    res = torch.max(-self.memory_spike,self.mem_pot)
                    res_spike_num = (res.abs()/(Vth_burst+1e-5)).floor() * torch.sign(res)
                    res_spike_num = torch.clamp(res_spike_num, -self.burst_T, self.burst_T)
                    res_spike = res_spike_num * Vth_burst
                    self.mem_pot -= res_spike
                    self.memory_spike += res_spike
                    spike += res_spike
                else:
                    res = torch.clamp(self.mem_pot,0.)
                    res_spike_num = (res/(Vth_burst+1e-5)).floor() * 1.0
                    res_spike_num = torch.clamp(res_spike_num, 0., self.burst_T)
                    res_spike = res_spike_num * Vth_burst
                    self.mem_pot -= res_spike
                    spike += res_spike
                    
            self.t = (self.t+1) % self.T
            return spike
        
        elif self.use_spike and isinstance(self.relu, StraightThrough):
            return self.relu(self.fwd_func(input, self.org_weight, self.org_bias))
        
        else:
            return self.relu(self.fwd_func(input, self.org_weight, self.org_bias))

    def init_module(self):
        self.mem_pot = self.mem_pot_init if isinstance(self.mem_pot_init, int) else self.mem_pot_init.clone()
        self.memory_spike = 0
        self.t = 0


def snn_qk_product(sx_t,sum_q,sum_k,q_proj_weight,k_proj_weight, q_proj_bias,k_proj_bias,num_heads,T,value_scale_factor=1.):
    bsz = sx_t.shape[1]
    seq_len = sx_t.shape[0]
    hidden_dim = q_proj_weight.shape[0]
    head_dim = hidden_dim//num_heads 
    sq_t = sx_t @ q_proj_weight.t() + q_proj_bias # [seq_len, batch_size, hidden_dim]
    sk_t = sx_t @ k_proj_weight.t() + k_proj_bias  # [seq_len, batch_size, hidden_dim]

    sq_t = sq_t.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1) # [num_head*bsz, seq_len, head_dim]
    sk_t = sk_t.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    sum_q += sq_t 
    sp_t = sum_q @ sk_t.transpose(1,2) + sq_t @ sum_k.transpose(1,2)
    sp_t /= math.sqrt(sk_t.shape[-1])
    sp_t *= value_scale_factor
    sum_k += sk_t
    return sum_q,sum_k,sp_t


def ann_qk_product(x,q_proj_weight,k_proj_weight,q_proj_bias,k_proj_bias,num_heads):
    q = x @ q_proj_weight.t() + q_proj_bias
    k = x @ k_proj_weight.t() + k_proj_bias
    seq_len = x.shape[0]
    bsz = x.shape[1]
    hidden_dim = q_proj_weight.shape[0]
    head_dim = hidden_dim//num_heads
    q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1) #[b,seq_len,feat_num]
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) #[b,seq_len,feat_num]
    p = q @ k.transpose(1,2)
    p /= math.sqrt(k.shape[-1])
    return p


class SpikeMultiheadAttention(nn.Module):

    def __init__(self, module: nn.Module, T:int = None, **kwargs):
        # module = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first, device=DEVICE)
        super(SpikeMultiheadAttention, self).__init__()
        self.T = kwargs.get('T', 32)
        self.product = SpikeProduct(T=self.T,module=module)
        self.spike_x2x = x2x_to_spike_module(X2X().to(next(module.parameters()).device),self.T)
        self.spike_x2x_pos = x2x_pos_to_spike_module(X2X_POS().to(next(module.parameters()).device),self.T)
        self.q_proj_weight,self.k_proj_weight,self.v_proj_weight = module.in_proj_weight.chunk(3)
        self.q_proj_bias,self.k_proj_bias,self.v_proj_bias = module.in_proj_bias.chunk(3)
        self.out_proj_weight = module.out_proj.weight
        self.out_proj_bias = module.out_proj.bias
        self.use_spike = False
        self.num_heads = module.num_heads
        self.sum_input = 0
        self.sum_output = 0
        self.t = 0
        self.output_encoder = True
        self.bipolar_with_memory = kwargs.get('bipolar_with_memory', False)
        self.burst_T = kwargs.get('burst_T', 2)
        for n,m in self.named_modules():
            # print(f"n:{n} -> m:{m}")
            if isinstance(m,SpikeLinear_ReLU) and not isinstance(m.relu,StraightThrough):
                m.bipolar_with_memory = self.bipolar_with_memory
                m.burst_T = self.burst_T

    def forward(self, input: torch.Tensor, k: torch.Tensor, v: torch.Tensor, need_weights = False, attn_mask = None, output_attentions = False):
        bsz = input.shape[1]
        seq_len = input.shape[0]
        embed_size = input.shape[2]
        hidden_dim = self.q_proj_weight.shape[0]
        head_dim = hidden_dim//self.num_heads
        device = input.device
        if self.use_spike: # snn
            if self.t == 0:
                self.sum_p = torch.zeros(self.num_heads * bsz, seq_len, seq_len).to(device)
                self.sum_m = 0
                self.sum_p2 = 0
                self.mem_pot = 0
                self.sum_p2_corr = 0
                self.sum_sum_p = 0
                self.sum_m_pre = 0
            
            ## QK product ##
            p = self.product(input)
            self.sum_p += p
            v = input @ self.v_proj_weight.t() + self.v_proj_bias
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
            
            ## softmax ##
            mean_p = self.sum_p/(self.t+1)**2
            m = fit_softmax(mean_p,dim=-1)
            m = (self.t+1) * m - self.sum_m
            sm = self.spike_x2x_pos(m)
            
            ## (QK)V product ##
            self.sum_m += sm
            if self.t == 0: self.sum_v = torch.zeros_like(v)
            p2 = self.sum_m @ v + sm @ self.sum_v
            self.sum_v += v
            self.sum_p2 += p2
            p2_corr = (self.t+1) * self.sum_p2/(self.t+1)**2 - self.sum_p2_corr
            self.sum_p2_corr += p2_corr
            output = p2_corr.transpose(0, 1).contiguous().view(seq_len, bsz, embed_size)
            if self.output_encoder:
                output = self.spike_x2x(output) # float to spikes 
            
            ## out projection ##
            output = output @ self.out_proj_weight.t() + self.out_proj_bias
            self.t = (self.t+1) % self.T
            return (output,None)

        else: # ann
            p = self.product(input)
            v = input @ self.v_proj_weight.t() + self.v_proj_bias
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
            m = fit_softmax(p,dim=-1)
            m = self.spike_x2x_pos(m)     
            output = m @ v
            output = output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_size)
            if self.output_encoder:
                output = self.spike_x2x(output) # float to spikes
            output = output @ self.out_proj_weight.t() + self.out_proj_bias
            return (output,None)
    def init_module(self):
        self.t = 0
        self.product.init_module()


class SpikeProduct(nn.Module):

    def __init__(self, T: int, module: nn.MultiheadAttention):
        super(SpikeProduct, self).__init__()
        self.T = T
        self.value_scale_factor = 1.
        self.in_proj_weight = module.in_proj_weight
        self.in_proj_bias = module.in_proj_bias
        self.q_proj_weight,self.k_proj_weight,self.v_proj_weight = self.in_proj_weight.chunk(3)
        self.q_proj_bias,self.k_proj_bias,self.v_proj_bias = self.in_proj_bias.chunk(3)
        self.use_spike = False
        self.num_heads = module.num_heads
        self.sum_q = None
        self.sum_k = None
        self.t = 0

    def forward(self, input: torch.Tensor):
        bsz = input.shape[1]
        num_heads = self.num_heads
        seq_len = input.shape[0]
        hidden_dim = self.q_proj_weight.shape[0]
        head_dim = hidden_dim//num_heads
        device = input.device
        if self.use_spike: # snn
            if self.t == 0:
                self.sum_q = torch.zeros(num_heads * bsz, seq_len, head_dim).to(device)
                self.sum_k = torch.zeros(num_heads * bsz, seq_len, head_dim).to(device)
                self.sum_p = torch.zeros(num_heads * bsz, seq_len, seq_len).to(device)
            self.sum_q,self.sum_k,sp_t = snn_qk_product(input,self.sum_q,self.sum_k,self.q_proj_weight,self.k_proj_weight, self.q_proj_bias,self.k_proj_bias,num_heads,self.T,self.value_scale_factor)
            self.t = (self.t+1) % self.T  
            return sp_t
        else: # ann
            p = ann_qk_product(input,self.q_proj_weight,self.k_proj_weight,self.q_proj_bias,self.k_proj_bias,num_heads)
            return p
    def init_module(self):
        self.t = 0


class STARobertaSelfAttention(nn.Module):
    def __init__(self, module: nn.Module, T:int = None, **kwargs):
        super(STARobertaSelfAttention, self).__init__()
        # Basic Vars
        self.num_attention_heads = module.num_attention_heads
        self.attention_head_size = module.attention_head_size
        self.all_head_size = module.all_head_size

        self.query = module.query
        self.key = module.key
        self.value = module.value

        self.dropout = module.dropout
        self.position_embedding_type = module.position_embedding_type

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = module.max_position_embeddings
            self.distance_embedding = module.distance_embedding

        self.is_decoder = module.is_decoder
        # STA
        self.T = T if T is not None else kwargs.get('T', 32)
        self.spike_x2x =        x2x_to_spike_module(X2X().to(next(module.parameters()).device),self.T)
        self.spike_x2x_pos =    x2x_pos_to_spike_module(X2X_POS().to(next(module.parameters()).device),self.T)
        self.sum_input  = 0
        self.sum_output = 0
        self.t = 0
        self.output_encoder = True
        self.use_spike      = False
        self.bipolar_with_memory = kwargs.get('bipolar_with_memory', False)
        self.value_scale_factor = 1.
        self.burst_T = kwargs.get('burst_T', 2)
        for n,m in self.named_modules():
            if isinstance(m,SpikeLinear_ReLU) and not isinstance(m.relu,StraightThrough):
                m.bipolar_with_memory = self.bipolar_with_memory
                m.burst_T = self.burst_T

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        bsz = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        embed_size = hidden_states.shape[2]
        hidden_dim = self.query.weight.shape[0]
        head_dim = hidden_dim // self.num_attention_heads
        device = hidden_states.device
        if self.use_spike: # snn

            if self.self_att_t == 0:
                self.sum_p = torch.zeros(self.num_attention_heads * bsz, seq_len, seq_len).to(device)
                self.sum_m = 0
                self.sum_p2 = 0
                self.mem_pot = 0
                self.sum_p2_corr = 0
                self.sum_sum_p = 0
                self.sum_m_pre = 0
            is_cross_attention = encoder_hidden_states is not None

            if is_cross_attention and past_key_value is not None:
                # reuse k,v, cross_attentions
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
                attention_mask = encoder_attention_mask
            elif is_cross_attention:
                key_layer = self.key(encoder_hidden_states)
                value_layer = self.value(encoder_hidden_states)
                attention_mask = encoder_attention_mask
            elif past_key_value is not None:
                key_layer = self.key(hidden_states)
                value_layer = self.value(hidden_states)
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self.key(hidden_states)
                value_layer = self.value(hidden_states)

            query_layer = mixed_query_layer

            

            use_cache = past_key_value is not None
            if self.is_decoder:
                past_key_value = (self.transpose_for_scores(key_layer), self.transpose_for_scores(value_layer))

            # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            ## QK^T product ##
            if self.prod_t == 0:
                self.prod_sum_q = torch.zeros(self.num_attention_heads * bsz, seq_len, head_dim).to(device)
                self.prod_sum_k = torch.zeros(self.num_attention_heads * bsz, seq_len, head_dim).to(device)
                self.prod_sum_p = torch.zeros(self.num_attention_heads * bsz, seq_len, seq_len).to(device)

            sq_t    = query_layer.contiguous().view(seq_len, bsz * self.num_attention_heads, head_dim).transpose(0, 1) # [num_head*bsz, seq_len, head_dim]
            sk_t    = key_layer.contiguous().view(-1, bsz * self.num_attention_heads, head_dim).transpose(0, 1)
            self.prod_sum_q += sq_t 
            sp_t = self.prod_sum_q @ sk_t.transpose(1,2) + sq_t @ self.prod_sum_k.transpose(1,2)
            sp_t /= math.sqrt(sk_t.shape[-1])
            sp_t *= self.value_scale_factor
            self.prod_sum_k += sk_t
            self.prod_t = (self.prod_t+1) % self.T  
            self.sum_p += sp_t
            ## softmax ##
            attention_scores = self.sum_p/(self.t+1)**2
            attention_scores = attention_scores.view(bsz, self.num_attention_heads, seq_len, seq_len)


            if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
                query_length, key_length = query_layer.shape[2], key_layer.shape[2]
                if use_cache:
                    position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                        -1, 1
                    )
                else:
                    position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
                position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
                distance = position_ids_l - position_ids_r

                positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
                positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

                if self.position_embedding_type == "relative_key":
                    relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores
                elif self.position_embedding_type == "relative_key_query":
                    relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            # Normalize the attention scores to probabilities.
            convert_back_attention_scores = attention_scores.view(self.num_attention_heads * bsz, seq_len, seq_len)
            
            attention_probs = fit_softmax(convert_back_attention_scores, dim=-1)
            attention_probs = attention_probs.view(bsz, self.num_attention_heads, seq_len, seq_len)
            attention_probs = self.dropout(attention_probs)
            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            attention_probs = attention_probs.view(bsz * self.num_attention_heads, seq_len, seq_len)
            attention_probs = (self.self_att_t+1) * attention_probs - self.sum_m
            sm = self.spike_x2x_pos(attention_probs)
            
            ## (QK)V product ##
            self.sum_m += sm
            value_layer = value_layer.contiguous().view(-1, bsz * self.num_attention_heads, head_dim).transpose(0, 1)

            if self.self_att_t == 0: 
                self.sum_v = torch.zeros_like(value_layer)
            p2 = self.sum_m @ value_layer + sm @ self.sum_v
            
            self.sum_v += value_layer
            self.sum_p2 += p2
            p2_corr = (self.self_att_t+1) * self.sum_p2/(self.self_att_t+1)**2 - self.sum_p2_corr
            self.sum_p2_corr += p2_corr
            output = p2_corr.transpose(0, 1).contiguous().view(seq_len, bsz, embed_size)
            if self.output_encoder:
                context_layer = self.spike_x2x(output) # float to spikes 
            
            context_layer = context_layer.view(bsz, self.num_attention_heads, seq_len, self.attention_head_size)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            self.self_att_t = (self.self_att_t+1) % self.T

            if self.is_decoder:
                outputs = outputs + (past_key_value,)
            return outputs        
        else: # ann
            # If this is instantiated as a cross-attention module, the keys
            # and values come from an encoder; the attention mask needs to be
            # such that the encoder's padding tokens are not attended to.
            is_cross_attention = encoder_hidden_states is not None

            if is_cross_attention and past_key_value is not None:
                # reuse k,v, cross_attentions
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
                attention_mask = encoder_attention_mask
            elif is_cross_attention:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
                attention_mask = encoder_attention_mask
            elif past_key_value is not None:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

            query_layer = self.transpose_for_scores(mixed_query_layer)
            # print("normal mode")
            # print("hidden_states", hidden_states.shape)
            use_cache = past_key_value is not None
            if self.is_decoder:
                # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                # Further calls to cross_attention layer can then reuse all cross-attention
                # key/value_states (first "if" case)
                # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                # all previous decoder key/value_states. Further calls to uni-directional self-attention
                # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                # if encoder bi-directional self-attention `past_key_value` is always `None`
                past_key_value = (key_layer, value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            # print("attention_scores", attention_scores.shape)
            if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
                query_length, key_length = query_layer.shape[2], key_layer.shape[2]
                if use_cache:
                    position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                        -1, 1
                    )
                else:
                    position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
                position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
                distance = position_ids_l - position_ids_r

                positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
                positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

                if self.position_embedding_type == "relative_key":
                    relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores
                elif self.position_embedding_type == "relative_key_query":
                    relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
                attention_scores = attention_scores + attention_mask
            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            attention_probs = self.spike_x2x_pos(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer_original_shape = context_layer.shape
            context_layer.view(seq_len, bsz, embed_size)
            if self.output_encoder:
                context_layer = self.spike_x2x(context_layer) # float to spikes 
            context_layer = context_layer.view(context_layer_original_shape)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            if self.is_decoder:
                outputs = outputs + (past_key_value,)
            return outputs
    
    def init_module(self):
        self.prod_t = 0
        self.self_att_t = 0



class STARobertaAttention(nn.Module):
    def __init__(self, module: nn.Module, T:int = None, **kwargs):
        super(STARobertaAttention, self).__init__()
        self.self = STARobertaSelfAttention(module.self, T=T, **kwargs)
        self.output = module.output
        self.pruned_heads = set()
        self.use_spike = False
        self.output_encoder = True

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self.self.use_spike = self.use_spike
        self.self.output_encoder = self.output_encoder

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def init_module(self):
        self.self.init_module()


class LinearLN(nn.Module):
    def __init__(self, embed_size: int, hidden_neuron: int, n_layer: int):
        super(LinearLN, self).__init__()
        self.embed_size = embed_size
        self.hidden_neuron = hidden_neuron
        self.n_layer = n_layer
        self.sequential = nn.Sequential(*(
            [nn.Linear(self.embed_size, self.hidden_neuron),nn.ReLU()]
            +[elem for sublst in [[nn.Linear(self.hidden_neuron, self.hidden_neuron),nn.ReLU()] for i in range(self.n_layer-1)] for elem in sublst]
            +[nn.Linear(self.hidden_neuron, self.embed_size)]
        ))

    def forward(self, input: torch.Tensor):
        output = self.sequential(input)
        return output


class SpikeLN(nn.Module):
    def __init__(self, module: nn.Module, T: int = None, **kwargs):
        super(SpikeLN, self).__init__()
        # module = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=DEVICE)
        self.module = module
        self.T = T if T is not None else kwargs.get('T', 32)
        self.use_spike = False
        self.t = 0
        self.gamma = module.weight
        self.beta = module.bias
        self.spike_sqrtinv = sqrtinv_to_spike_module(sqrtinv,self.T)
        self.spike_x2x = x2x_to_spike_module(X2X().to(next(module.parameters()).device), self.T, belong_to_ln=True)
        self.output_encoder = True
        self.bipolar_with_memory = kwargs.get('bipolar_with_memory', False)
        self.burst_T = kwargs.get('burst_T', 2)
        for n,m in self.named_modules():
            if isinstance(m,SpikeLinear_ReLU) and not isinstance(m.relu,StraightThrough):
                m.bipolar_with_memory = self.bipolar_with_memory
                m.burst_T = self.burst_T

    def forward(self, input: torch.Tensor):
        if self.use_spike: # snn
            n = input.shape[-1]
            W_rmvmean = torch.full((n,n), -1/n).to(device=input.device)
            W_rmvmean.fill_diagonal_(1-1/n)
            W_var = torch.full((n,1), 1/n).to(device=input.device)
            rmvmean = input @ W_rmvmean

            if self.t == 0:
                self.sum_var = 0
                self.sum_var_corr = 0
                self.sum_prod = 0
                self.sum_prod_corr = 0

            ## get variance ##
            self.sum1 += rmvmean
            vars = self.sum1 * rmvmean + rmvmean * self.sum2 
            self.sum2 += rmvmean
            var = vars @ W_var 
            self.sum_var += var
            var_corr = (self.t+1) * self.sum_var / (self.t+1)**2 - self.sum_var_corr # no corr: var_corr = var/self.T
            self.sum_var_corr += var_corr

            ## get inverse ##
            inverse = self.spike_sqrtinv(var_corr)

            ## get scalar product ##
            prod = self.sum2 * inverse + rmvmean * self.sum3
            self.sum3 += inverse
            self.sum_prod += prod
            prod_corr = (self.t+1) * self.sum_prod / (self.t+1)**2 - self.sum_prod_corr # no corr: prod_corr = prod/self.T
            self.sum_prod_corr += prod_corr
            if self.output_encoder:
                prod_corr = self.spike_x2x(prod_corr) # float to spikes

            # scale #
            norm = prod_corr * self.gamma + self.beta

            self.t = (self.t+1) % self.T
            return norm
        else: # ann
            n = input.shape[-1]
            W_rmvmean = torch.full((n,n), -1/n).to(device=input.device)
            W_rmvmean.fill_diagonal_(1-1/n)
            W_var = torch.full((n,1), 1/n).to(device=input.device)
            rmvmean = input @ W_rmvmean
            vars = rmvmean * rmvmean
            var = vars @ W_var
            reciprocal = self.spike_sqrtinv(var)
            prod = rmvmean * reciprocal
            if self.output_encoder:
                prod = self.spike_x2x(prod) # float to spikes
            norm = prod * self.gamma + self.beta
            return norm
    
    def init_module(self):
        self.sum1 = 0
        self.sum2 = 0
        self.sum3 = 0
        self.sum_delta = 0
        self.sum_prod = 0
        self.sum_output = 0
        self.record_spike = None
        self.t = 0


    
def TransformRelu (module: nn.Module, **kwargs):
    for name, immediate_child_module in reversed(list(module.named_children())):
        if isinstance(immediate_child_module, nn.ReLU):
            copy_relu = copy.deepcopy(immediate_child_module)
            copy_name = name

        if copy_relu is not None and isinstance(immediate_child_module,SpikeLinear_ReLU) and isinstance(immediate_child_module.relu,StraightThrough):
            immediate_child_module.add_module('relu', copy_relu)
            immediate_child_module.bipolar_with_memory = kwargs.get('bipolar_with_memory', False)
            immediate_child_module.burst_T = kwargs.get('burst_T', 32)
            setattr(module, copy_name, StraightThrough())
    return module

#             elif isinstance(immediate_child_module, (nn.ReLU, nn.ReLU6)):
#                 if prev_module is not None: # nn.Linear
#                     prev_module.add_module('relu', immediate_child_module)
#                     setattr(module, name, StraightThrough())
#                     prev_module.bipolar_with_memory = self.bipolar_with_memory
#                     prev_module.burst_T = self.burst_T
#                 else:


def set_spike_state(model: nn.Module, use_spike: bool = True):
    model.use_spike = use_spike
    for m in model.modules():
        if isinstance(m, SpikeLinear_ReLU):
            m.use_spike = use_spike
        if isinstance(m, SpikeLN):
            m.use_spike = use_spike
        if isinstance(m, SpikeMultiheadAttention):
            m.use_spike = use_spike
            m.product.use_spike = use_spike
        if isinstance(m, STARobertaAttention):
            m.self.use_spike = use_spike
            m.use_spike = use_spike


@torch.no_grad()
def get_maximum_activation(train_loader: Union[torch.utils.data.DataLoader,torch.Tensor],
                           model: nn.Module,
                           momentum: Union[float, None] = 0.9,
                           iters: int = 20,
                           T: int = 8,
                           mse: bool = True, 
                           percentile: Union[float, None] = None,
                           neuron_wise: bool = False,
                           dist_avg: bool = False):
    set_spike_state(model, use_spike=False)
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hook_list = []
    for n,m in model.named_modules():
        if isinstance(m, SpikeLinear_ReLU) and not isinstance(m.relu, StraightThrough):
            hook_list += [m.register_forward_hook(DataSaverHook(momentum, T, mse, percentile, neuron_wise=neuron_wise, dist_avg=dist_avg,name=n))]
    if isinstance(train_loader,torch.Tensor):
        for input in train_loader:
            input = input.to(device=device)
            _ = model(input)
        for h in hook_list:
            h.remove()
    else:
        # batch_elem_len = len(train_loader._get_iterator().next()) 
        batch_elem_len = len(next(iter(train_loader)))
        #Modified for Roberta

        if batch_elem_len == 3:
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = model(
                        **batch
                     )
                if step >= iters + 1:
                    break
        for h in hook_list:
            h.remove()
    set_spike_state(model, use_spike=True)


def init_sta_converted_model(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (SpikeLinear_ReLU, SpikeMultiheadAttention, STARobertaAttention, SpikeLN)):
            m.init_module()



def clip_floor(tensor: torch.Tensor, T: int, Vth: Union[float, torch.Tensor], shift: float = 0.0):
    snn_out = torch.clamp(tensor / Vth * T, min=0, max=T)
    return snn_out.floor() * Vth / T + shift * Vth / T


def quantized_qk_product(x,q_proj_weight,k_proj_weight, q_proj_bias,k_proj_bias,num_heads=12,T=16):
    q = x @ q_proj_weight.t() + q_proj_bias
    k = x @ k_proj_weight.t() + k_proj_bias
    seq_len = x.shape[0]
    bsz = x.shape[1]
    hidden_dim = q_proj_weight.shape[0]
    head_dim = hidden_dim//num_heads
    q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1) #[b,seq_len,feat_num]
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) #[b,seq_len,feat_num]
    p = q @ k.transpose(1,2)
    p /= math.sqrt(k.shape[-1])
    return p


class DataSaverHook:
    def __init__(self, momentum: Union[float, None] = 0.9, T: int = 8, mse: bool = True, percentile: Union[float, None] = None, 
                 neuron_wise: bool = False, dist_avg: bool = False, name=''):
        self.momentum = momentum
        self.max_act = None
        self.T = T
        self.mse = mse
        self.percentile = percentile
        self.neuron_wise = neuron_wise
        self.dist_avg = dist_avg
        self.name = name
        
    def __call__(self, module, input_batch, output_batch):

        def get_act_thresh(tensor):
            if self.mse:
                act_thresh = find_threshold_mse(output_batch, T=self.T, neuron_wise=self.neuron_wise)
            elif self.percentile is not None:
                assert 0. <= self.percentile <= 1.0
                act_thresh = quantile(output_batch, self.percentile)
            else:
                act_thresh = tensor.max()
            return act_thresh
        if self.max_act is None:
            self.max_act = get_act_thresh(output_batch)
        else:
            cur_max = get_act_thresh(output_batch)
            if self.momentum is None:
                self.max_act = self.max_act if self.max_act > cur_max else cur_max
            else:
                self.max_act = self.momentum * self.max_act + (1 - self.momentum) * cur_max
            
        if self.dist_avg:
            allaverage(self.max_act)
        
        module.threshold = self.max_act


def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p * 100)).type_as(tensor)


def find_threshold_mse(tensor: torch.Tensor, T: int = 8, neuron_wise: bool = False, iters: int = 95):
    max_act = tensor.max()
    best_score = 1e5
    best_Vth = 0
    if neuron_wise:
        if len(tensor.shape) == 3: # LND: D
            num_neuron = tensor.shape[-1]
            max_act = tensor.max(dim=0)[0].max(dim=0)[0].reshape(1, 1,num_neuron)
            best_score = torch.ones_like(max_act).mul(1e10)
            best_Vth = torch.clone(max_act)
            for i in range(95):
                new_Vth = max_act * (1.0 - (i * 0.005))
                mse = lp_loss(tensor, clip_floor(tensor, T, new_Vth), p=2.0, reduction='channel_split',dim=(0,1))
                mse = mse.reshape(1, 1,num_neuron)
                mask = mse < best_score
                best_score[mask] = mse[mask]
                best_Vth[mask] = new_Vth[mask]

        if len(tensor.shape) == 4: # LNDG: DG
            max_act = tensor.max(dim=0)[0].max(dim=0)[0].unsqueeze(0).unsqueeze(0)
            best_score = torch.ones_like(max_act).mul(1e10)
            best_Vth = torch.clone(max_act)
            for i in range(iters):
                new_Vth = max_act * (1.0 - (i * 0.005))
                mse = lp_loss(tensor, clip_floor(tensor, T, new_Vth), p=2.0, reduction='channel_split',dim=(0,1))
                mse = mse.reshape(max_act.shape)
                mask = mse < best_score
                best_score[mask] = mse[mask]
                best_Vth[mask] = new_Vth[mask]
            
    else:
        for i in range(iters):
            new_Vth = max_act * (1.0 - (i * 0.01))
            mse = lp_loss(tensor, clip_floor(tensor, T, new_Vth), p=2.0, reduction='other')
            if mse < best_score:
                best_Vth = new_Vth
                best_score = mse

    return best_Vth


def lp_loss(pred, tgt, p=2.0, reduction='none',dim=(0,1)):
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    elif reduction == 'channel_split':
        return (pred - tgt).abs().pow(p).sum(dim)
    else:
        return (pred - tgt).abs().pow(p).mean()




# ===============================================================================
#                             FIT NON-LINEAR
# ===============================================================================
    

class Distilled_EXP(nn.Module):
	def __init__(self, load_distilled_weights=False,num_neurons=32,path=None):
		super().__init__()
		self.approximator = nn.Sequential(
			nn.Linear(1, num_neurons),
			nn.ReLU(),
			nn.Linear(num_neurons, 1)
		)
		if load_distilled_weights:
			self.load_state_dict(torch.load(path))

	def forward(self,x):
		dim = x.dim()
		if dim == 0:
			return self.approximator(x)
		else:
			return torch.squeeze(self.approximator(torch.unsqueeze(x, -1)))


class Distilled_INV(nn.Module):
	def __init__(self, load_distilled_weights=False,num_neurons=32,path=None):
		super().__init__()
		self.approximator = nn.Sequential(
			nn.Linear(1, num_neurons),
			nn.ReLU(),
			nn.Linear(num_neurons, 1)
		)
		if load_distilled_weights:
			self.load_state_dict(torch.load(path))

	def forward(self,x):
		dim = x.dim()
		if dim == 0:
			return self.approximator(x)
		else:
			return torch.squeeze(self.approximator(torch.unsqueeze(x, -1)),dim=-1)
        


class Distilled_GELU(nn.Module):
    def __init__(self, load_distilled_weights=False,num_neurons=64, path = None):
        super().__init__()
        self.approximator = nn.Sequential(
            nn.Linear(1, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 1)
            )
        self.approximator.requires_grad = False
        
        if load_distilled_weights:
            self.load_state_dict(torch.load(path))
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        dim = x.dim()
        if dim == 0:
            return self.approximator(x)
        else:
            return torch.squeeze(self.approximator(torch.unsqueeze(x, -1)))


class Distilled_SQRTINV(nn.Module):
    def __init__(self, load_distilled_weights=False,num_neurons=8,path = None):
        super().__init__()
        self.approximator = nn.Sequential(
            nn.Linear(1, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons,1),
        )
        self.approximator.requires_grad = False 

        if load_distilled_weights:
            self.load_state_dict(torch.load(path))
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.approximator(x)


def get_distilled_exp(device='cuda', float16=True, num_neurons=32, path=None):
	if float16:
		return Distilled_EXP(load_distilled_weights = True,num_neurons=num_neurons,path=path).half().to(device)
	return Distilled_EXP(load_distilled_weights = True, num_neurons=num_neurons,path=path).to(device)


def get_distilled_inv(device='cuda', float16=True, num_neurons=32, path=None):
	if float16:
		return Distilled_INV(load_distilled_weights=True,num_neurons=num_neurons,path=path).half().to(device)
	return Distilled_INV(load_distilled_weights=True,num_neurons=num_neurons,path=path).to(device)


def get_distilled_sqrtinv(device='cuda', float16=True, num_neurons=64, path=None):
	if float16:
		return Distilled_SQRTINV(load_distilled_weights=True,num_neurons=num_neurons,path=path).half().to(device)
	return Distilled_SQRTINV(load_distilled_weights=True,num_neurons=num_neurons,path=path).to(device)


def get_distilled_gelu(device='cuda', float16=True, num_neurons=8, path=None):
	if float16:
		return Distilled_GELU(load_distilled_weights=True,num_neurons=num_neurons,path=path).half().to(device)
	return Distilled_GELU(load_distilled_weights=True,num_neurons=num_neurons,path=path).to(device)


def replace_gelu_with_relu(model,convert_layers,num_neurons=64,path=GELU_PATH,device='cpu'):
    for m_str in convert_layers:
        eval(f'model.{m_str}').mlp.gelu = get_distilled_gelu(float16=False,device=DEVICE,num_neurons=num_neurons,path=path)


sqrtinv = get_distilled_sqrtinv(float16=False,device=DEVICE,num_neurons=8,path=SQRTINV_PATH)
exper = get_distilled_exp(float16=False,device=DEVICE,num_neurons=32,path=EXP_PATH)
inver = get_distilled_inv(float16=False,device=DEVICE,num_neurons=64,path=INV_PATH)


def fit_softmax(X,dim=-1):
    ## data translation ##
    tmax2=X[:,:,0].unsqueeze(-1)
    tmax=tmax2
    tp=X-(tmax-1)

    ## data clamp ##
    tp[tp>3]=3
    index=[tp>-20]
    X_up=torch.zeros_like(tp)
    X_up[index]=exper(tp[index])

    partition=X_up.sum(dim=dim,keepdim=True)
    p_inv = inver(partition)
    out = X_up * p_inv

    ## re-norm ##
    partition_p2=out.sum(dim=dim,keepdim=True)
    index=(partition_p2>1.5).squeeze()
    p_inv2 = inver(partition_p2)
    out_2=torch.zeros_like(out)
    out_2[~index]=out[~index]
    out_2[index]=out[index]*p_inv2[index]
    return out_2


def sqrtinv_to_spike_module(ann_module,T):
    snn_module = copy.deepcopy(ann_module)
    snn_module.approximator[0]

    snn_module.approximator[0] = SpikeLinear_ReLU(module=ann_module.approximator[0], T=T)
    snn_module.approximator[0].relu = nn.ReLU()
    snn_module.approximator[0].belong_to_ln = True
    snn_module.approximator[1] = StraightThrough()
    snn_module.approximator[2] = SpikeLinear_ReLU(module=ann_module.approximator[2], T=T)
    return snn_module




# ===============================================================================
#                               SPIKE ENCODER
# ===============================================================================


class X2X(nn.Module):
    def __init__(self):
        super().__init__()
        self.approximator = nn.Sequential(
            nn.Linear(1, 2, bias=False),
            nn.ReLU(),
            nn.Linear(2, 1, bias=False)
            )
        self.approximator.requires_grad = False
        self.approximator[0].weight.data[0,0] = 1.0
        self.approximator[0].weight.data[1,0] = -1.0
        self.approximator[2].weight.data[0,0] = 1.0
        self.approximator[2].weight.data[0,1] = -1.0
    
    def forward(self, x):
        dim = x.dim()
        if dim == 0:
            return self.approximator(x)
        else:
            return torch.squeeze(self.approximator(torch.unsqueeze(x, -1)))


class X2X_POS(nn.Module):
    def __init__(self):
        super().__init__()
        self.approximator = nn.Sequential(
            nn.Linear(1, 1, bias=False),
            nn.ReLU(),
            )
        self.approximator.requires_grad = False
        self.approximator[0].weight.data[0,0] = 1.0
    
    def forward(self, x):
        dim = x.dim()
        if dim == 0:
            return self.approximator(x)
        else:
            return torch.squeeze(self.approximator(torch.unsqueeze(x, -1)))


def x2x_to_spike_module(ann_module,T,belong_to_ln=False):
    snn_module = copy.deepcopy(ann_module)
    snn_module.approximator[0]

    snn_module.approximator[0] = SpikeLinear_ReLU(module=ann_module.approximator[0], T=T)
    snn_module.approximator[0].relu = nn.ReLU()
    snn_module.approximator[0].belong_to_x2x = True
    snn_module.approximator[0].belong_to_ln = belong_to_ln
    snn_module.approximator[1] = StraightThrough()
    snn_module.approximator[2] = SpikeLinear_ReLU(module=ann_module.approximator[2], T=T)
    return snn_module


def x2x_pos_to_spike_module(ann_module,T):
    snn_module = copy.deepcopy(ann_module)
    snn_module.approximator[0]
    snn_module.approximator[0] = SpikeLinear_ReLU(module=ann_module.approximator[0], T=T)
    snn_module.approximator[0].relu = nn.ReLU()
    snn_module.approximator[0].belong_to_x2x_pos = True
    snn_module.approximator[1] = StraightThrough()
    return snn_module