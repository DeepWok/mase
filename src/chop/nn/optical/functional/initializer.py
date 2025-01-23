"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 01:57:16
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 01:57:18
"""
import numpy as np
import torch

__all__ = [
    "quant_kaiming_uniform",
    "quant_kaiming_uniform_",
    "truncated_normal",
    "truncated_normal_",
    "morr_uniform_",
    "morr_uniform",
]


def quant_kaiming_uniform(w, nbit, beta=1.5):
    """https://arxiv.org/pdf/1802.04680.pdf"""
    if w.dim() > 2:
        receptive_field = w[0, 0, ...].numel()
    else:
        receptive_field = 1
    fan_in = w.size(1) * receptive_field
    sigma = 2 ** (1 - nbit)
    L_min = beta * sigma
    L = max(np.sqrt(6 / fan_in), L_min)
    return w.clone().uniform_(-L, L)


def quant_kaiming_uniform_(w, nbit, beta=1.5):
    """https://arxiv.org/pdf/1802.04680.pdf"""
    if w.dim() > 2:
        receptive_field = w[0, 0, ...].numel()
    else:
        receptive_field = 1
    fan_in = w.size(1) * receptive_field
    sigma = 2 ** (1 - nbit)
    L = np.sqrt(6 / fan_in)
    L_min = beta * sigma
    scale = 2 ** round(np.log2(L_min / L))
    scale = max(scale, 1.0)
    L = max(L, L_min)

    return torch.nn.init.uniform_(w, -L, L), scale


def truncated_normal(tensor, mean=0, std=1, a=-2, b=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    a = (a - mean) / std
    b = (b - mean) / std
    valid = (tmp < b) & (tmp > a)
    ind = valid.max(-1, keepdim=True)[1]
    output = tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)
    return output


def truncated_normal_(tensor, mean=0, std=1, a=-2, b=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    a = (a - mean) / std
    b = (b - mean) / std
    valid = (tmp < b) & (tmp > a)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def morr_uniform_(tensor, MORRConfig, n_op=4, biased=False, gain=1):
    """
    description: Uniform initialization for MORR array based tensor core [SqueezeLight, Gu+, DATE'21]. We only consider how n_op influence one MORR's output. How to balance vector length should be considered in learnable balancing factor\\
    @tensor {torch.Tensor} weight tensor/parameter\\
    @MORRConfig {Config} MORR configuration defined in the onnlib/model/layer/device/mrr\\
    @n_op {int scalar} Number of operands on an MORR\\
    @biased {bool} biased=True, weight in [0, L]; otherwise in [-L/2, L/2].\\
    @gain {float} Gain due to activation. ReLU=sqrt(2), Tanh=5/3, Clamp(0,1)=2\\
    return {}
    """
    morr_fwhm = (
        -4
        * np.pi ** 2
        * MORRConfig.radius
        * MORRConfig.effective_index
        * (
            1 / MORRConfig.resonance_wavelength
            - 1 / (MORRConfig.resonance_wavelength - MORRConfig.bandwidth / 2)
        )
    )
    ### first we need to calculate the information gain of an MORR, estimated by linear estimation at 0 and FWHM
    # t1 = mrr_roundtrip_phase_to_tr_fused(torch.tensor([0]).float(), a=MORRConfig.attenuation_factor, r=MORRConfig.coupling_factor, intensity=True)
    # t2 = mrr_roundtrip_phase_to_tr_fused(torch.tensor([morr_fwhm]).float(), a=MORRConfig.attenuation_factor, r=MORRConfig.coupling_factor, intensity=True)
    # g = (t2 - t1) / morr_fwhm

    ### calculate the variance of the weight
    # var_phi = 1 ## assume the input is normalized to have variance 1
    # var_w = 1/(3/2*g**4*n_op*var_phi)

    ### calculate range of uniform distribution U(-L,L)
    # L = ((3 * var_w)**0.5).item()
    # return torch.nn.init.uniform_(tensor, -L, L)

    ## approximation by assuming 4*std(phi)= 3*FWHM, E[x]=0, D[x]=1, W ~ U[0, L]
    L = (3 / (4 * n_op)) ** 0.5 * morr_fwhm * gain
    if biased:
        return torch.nn.init.uniform_(tensor, 0, L)
    else:
        return torch.nn.init.uniform_(tensor, -L / 2, L / 2)


def morr_uniform(tensor, MORRConfig, n_op=4, biased=False, gain=1):
    """
    description: Uniform initialization for MORR array based tensor core [SqueezeLight, Gu+, DATE'21]\\
    @tensor {torch.Tensor} weight tensor/parameter\\
    @MORRConfig {Config} MORR configuration defined in the onnlib/model/layer/device/mrr\\
    @n_op {int scalar} Number of operands on an MORR\\
    @biased {bool} biased=True, weight in [0, L]; otherwise in [-L/2, L/2].\\
    @gain {float} Gain due to activation. ReLU=sqrt(2), Tanh=5/3, Clamp(0,1)=2\\
    return {}
    """
    morr_fwhm = (
        -4
        * np.pi ** 2
        * MORRConfig.radius
        * MORRConfig.effective_index
        * (
            1 / MORRConfig.resonance_wavelength
            - 1 / (MORRConfig.resonance_wavelength - MORRConfig.bandwidth / 2)
        )
    )
    ### first we need to calculate the information gain of an MORR, estimated by linear estimation at 0 and FWHM
    # t1 = mrr_roundtrip_phase_to_tr_fused(torch.tensor([0]), a=MORRConfig.attenuation_factor, r=MORRConfig.coupling_factor, intensity=True)
    # t2 = mrr_roundtrip_phase_to_tr_fused(torch.tensor([morr_fwhm]), a=MORRConfig.attenuation_factor, r=MORRConfig.coupling_factor, intensity=True)
    # g = (t2 - t1) / morr_fwhm

    # var_phi = 1 ## assume the input is normalized to have variance 1
    # var_w = 1/(3/2*g**4*n_op*var_phi)

    # ### calculate range of uniform distribution U(-L,L)
    # L = (3 * var_w)**0.5
    # return tensor.clone().uniform_(-L, L)

    ## approximation by assuming 4*std(phi)= 3*FWHM, E[x]=0, D[x]=1, W ~ U[0, L]
    L = (3 / (4 * n_op)) ** 0.5 * morr_fwhm * gain
    if biased:
        return tensor.clone().uniform_(0, L)
    else:
        return tensor.clone().uniform_(-L / 2, L / 2)
