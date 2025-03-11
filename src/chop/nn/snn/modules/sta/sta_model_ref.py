import copy
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from clip.model import QuickGELU
# from torch.autograd import Variable
# from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import random


DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"
INV_PATH = '/home/george/mase/src/chop/nn/snn/modules/sta/premodels/distilled_inv_64.pth'
EXP_PATH = '/home/george/mase/src/chop/nn/snn/modules/sta/premodels/distilled_exp_32.pth'
SQRTINV_PATH = '/home/george/mase/src/chop/nn/snn/modules/sta/premodels/distilled_sqrtinv_8.pth'
GELU_PATH = '/home/george/mase/src/chop/nn/snn/modules/sta/premodels/distilled_gelu_64.pth'

class Ref_StraightThrough(nn.Module):
    
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class Ref_SpikeLinear_ReLU(nn.Module):
    def __init__(self, T: int, module:nn.Module):
        super(Ref_SpikeLinear_ReLU, self).__init__()
        self.T = T
        self.t = 0
        self.threshold = None
        self.mem_pot = 0
        self.mem_pot_init = 0
        self.use_spike = False
        self.relu = Ref_StraightThrough()
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
        if self.use_spike and not isinstance(self.relu, Ref_StraightThrough):

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
        
        elif self.use_spike and isinstance(self.relu, Ref_StraightThrough):
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


class Ref_SpikeAttention(nn.Module):

    def __init__(self, T: int, module: nn.MultiheadAttention):
        super(Ref_SpikeAttention, self).__init__()
        self.T = T
        self.product = SpikeProduct(T=T,module=module)
        self.spike_x2x = x2x_to_spike_module(X2X().to(next(module.parameters()).device),T)
        self.spike_x2x_pos = x2x_pos_to_spike_module(X2X_POS().to(next(module.parameters()).device),T)
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

    def forward(self, input: torch.Tensor, k: torch.Tensor, v: torch.Tensor, need_weights = False, attn_mask = None):
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


class Ref_SpikeLN(nn.Module):
    def __init__(self, T: int, module: nn.LayerNorm):
        super(Ref_SpikeLN, self).__init__()
        self.module = module
        self.T = T
        self.use_spike = False
        self.t = 0
        self.gamma = module.weight
        self.beta = module.bias
        self.spike_sqrtinv = sqrtinv_to_spike_module(sqrtinv,self.T)
        self.spike_x2x = x2x_to_spike_module(X2X().to(next(module.parameters()).device),T,belong_to_ln=True)
        self.output_encoder = True

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



# ===============================================================================
#                                  SPIKE MODEL 
# ===============================================================================


# class SpikeModel(nn.Module):
#     def __init__(self, model: nn.Module, T: int, convert_layers = None, bipolar_with_memory=False, burst_T=0):
#         super().__init__()
#         self.model = model
#         self.use_spike = False
#         self.bipolar_with_memory = bipolar_with_memory
#         self.burst_T = burst_T
#         self.spike_model_refactor(model,T,convert_layers)
#         assert T > 0, "SNN does not accept negative simulation length"
#         self.T = T
    
#     def spike_model_refactor(self, module: nn.Module, T: int, convert_layers=None):
#         for m_str in convert_layers:
#             m = eval(f'module.{m_str}')
#             self.spike_module_refactor(m,T)

#     def spike_module_refactor(self, module: nn.Module, T: int, prev_module=None):
#         for name, immediate_child_module in module.named_children():
#             print("name",name)
#             print("immediate_child_module",immediate_child_module)
#             if isinstance(immediate_child_module,nn.LayerNorm):
#                 setattr(module,name,SpikeLN(T=T,module = immediate_child_module))
#                 prev_module = getattr(module, name)
#                 for n,m in prev_module.named_modules():
#                     if isinstance(m,SpikeLinear_ReLU) and not isinstance(m.relu,StraightThrough):
#                         m.bipolar_with_memory = self.bipolar_with_memory
#                         m.burst_T = self.burst_T
#                 pass
#             elif name == 'attn':
#                 print("immediate_child_module",immediate_child_module)
#                 setattr(module,name,SpikeAttention(T=T,module = immediate_child_module))
#                 prev_module = getattr(module, name)
#                 for n,m in prev_module.named_modules():
#                     if isinstance(m,SpikeLinear_ReLU) and not isinstance(m.relu,StraightThrough):
#                         m.bipolar_with_memory = self.bipolar_with_memory
#                         m.burst_T = self.burst_T
#                 pass
#             elif isinstance(immediate_child_module,nn.Linear):
#                 setattr(module,name,SpikeLinear_ReLU(T=T,module = immediate_child_module))
#                 prev_module = getattr(module, name)
#                 pass
#             elif isinstance(immediate_child_module, (nn.ReLU, nn.ReLU6)):
#                 if prev_module is not None: # nn.Linear
#                     prev_module.add_module('relu', immediate_child_module)
#                     setattr(module, name, StraightThrough())
#                     prev_module.bipolar_with_memory = self.bipolar_with_memory
#                     prev_module.burst_T = self.burst_T
#                 else:
#                     continue
#                 pass
            
#             else:
#                 prev_module = self.spike_module_refactor(
#                     immediate_child_module, T=T, prev_module=prev_module)
#         return prev_module
    
    def set_spike_state(self, use_spike: bool = True):
        self.use_spike = use_spike
        for m in self.model.modules():
            if isinstance(m, Ref_SpikeLinear_ReLU):
                m.use_spike = use_spike
            if isinstance(m, Ref_SpikeLN):
                m.use_spike = use_spike
            if isinstance(m, Ref_SpikeAttention):
                m.use_spike = use_spike
                m.product.use_spike = use_spike
            

    def init_model(self):
        for m in self.model.modules():
            if isinstance(m, (Ref_SpikeLinear_ReLU, Ref_SpikeAttention, Ref_SpikeLN)):
                m.init_module()

    def forward(self, input):
        if self.use_spike:
            self.init_model()
            out = 0
            for t in range(self.T):
                out_t = self.model(input)
                out += out_t
                torch.cuda.empty_cache()
                import gc 
                gc.collect()
        else:
            out = self.model(input)
        return out
    
# @torch.no_grad()
# def get_maximum_activation(train_loader: Union[torch.utils.data.DataLoader,torch.Tensor],
#                            model: SpikeModel,
#                            momentum: Union[float, None] = 0.9,
#                            iters: int = 20,
#                            T: int = 8,
#                            mse: bool = True, 
#                            percentile: Union[float, None] = None,
#                            neuron_wise: bool = False,
#                            dist_avg: bool = False):
#     model.set_spike_state(use_spike=False)
#     model.eval()
#     device = next(model.parameters()).device
#     dtype = next(model.parameters()).dtype
#     hook_list = []
#     for n,m in model.named_modules():
#         if isinstance(m, SpikeLinear_ReLU) and not isinstance(m.relu, StraightThrough):
#             hook_list += [m.register_forward_hook(DataSaverHook(momentum, T, mse, percentile, neuron_wise=neuron_wise, dist_avg=dist_avg,name=n))]
#     if isinstance(train_loader,torch.Tensor):
#         for input in train_loader:
#             input = input.to(device=device)
#             _ = model(input)
#         for h in hook_list:
#             h.remove()
#     else:
#         # batch_elem_len = len(train_loader._get_iterator().next()) 
#         batch_elem_len = len(next(iter(train_loader)))
#         if batch_elem_len == 2:
#             for i, (input, target) in enumerate(train_loader):
#                 print(f'{i}/{iters}')
#                 input = input.to(device=device).type(dtype)
#                 _ = model(input)
#                 if i >= iters:
#                     break
#         for h in hook_list:
#             h.remove()


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
    snn_module.approximator[0] = Ref_SpikeLinear_ReLU(T=T,module=ann_module.approximator[0])
    snn_module.approximator[0].relu = nn.ReLU()
    snn_module.approximator[0].belong_to_ln = True
    snn_module.approximator[1] = Ref_StraightThrough()
    snn_module.approximator[2] = Ref_SpikeLinear_ReLU(T=T,module=ann_module.approximator[2])
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
    snn_module.approximator[0] = Ref_SpikeLinear_ReLU(T=T,module=ann_module.approximator[0])
    snn_module.approximator[0].relu = nn.ReLU()
    snn_module.approximator[0].belong_to_x2x = True
    snn_module.approximator[0].belong_to_ln = belong_to_ln
    snn_module.approximator[1] = Ref_StraightThrough()
    snn_module.approximator[2] = Ref_SpikeLinear_ReLU(T=T,module=ann_module.approximator[2])
    return snn_module


def x2x_pos_to_spike_module(ann_module,T):
    snn_module = copy.deepcopy(ann_module)
    snn_module.approximator[0]
    snn_module.approximator[0] = Ref_SpikeLinear_ReLU(T=T,module=ann_module.approximator[0])
    snn_module.approximator[0].relu = nn.ReLU()
    snn_module.approximator[0].belong_to_x2x_pos = True
    snn_module.approximator[1] = Ref_StraightThrough()
    return snn_module