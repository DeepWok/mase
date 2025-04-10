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

from ..utils import MORRConfig_20um_MQ
from ..utils import mrr_roundtrip_phase_to_tr_func, mrr_roundtrip_phase_to_tr_fused
from ..utils import toeplitz
from ..utils import morr_uniform_
from ..utils import input_quantize_fn, weight_quantize_fn
from .base_layer import ONNBaseLayer
from .morr_custom_linear import AllPassMORRLinear
from .morr_linear import AllPassMORRCirculantLinear

from transformers import BertModel, BertForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
    Conv1D,
)

logger = logging.getLogger(__name__)

__all__ = ["AllPassMORRCirculantMatMals"]

def make_autoregressive_mask_for(x):
    length = x.size(1)
    ones = x.new_ones((length, length))
    mask = torch.triu(ones, diagonal=1) != 0.0
    return mask


def make_position_indices_for(x):
    length = x.size(1)
    batch_size = x.size(0)
    indices = torch.arange(length, device=x.device).repeat(batch_size, 1)
    return indices


def load_lookup_table(file, device):
    data = torch.from_numpy(numpy.genfromtxt(file, delimiter='\t')).float()
    levels = data.size(0)
    lower_bound = data[0,1].item()
    weight = data[:,1].unsqueeze(1).cuda(device)
    return weight, lower_bound, levels


def apply_lut_to_normalized(x, lut, bit_degredation=0):
    lut_weight, lut_lb, lut_levels = lut
    deg_factor = 2**bit_degredation
    x = x.mul(lut_levels - deg_factor).div(deg_factor).round().mul(deg_factor).to(dtype=torch.long)
    x = F.embedding(x, lut_weight).squeeze(-1)
    return x


class QuantizeValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quant_levels, min_val, max_val, quant_mode, lut_min=None):
        with torch.no_grad():
            diff = max_val - min_val
            x = x.clamp(min_val, max_val).add(-1.0 * min_val).div(diff + 1e-8).mul(quant_levels - 1)

            if quant_mode == 'det':
                x = x.round()
                x = x.div(quant_levels - 1).mul(diff).add(min_val)
            elif quant_mode == 'rand':
                x = x.add(torch.rand_like(x).add(-0.5)).round() # randn* 0.288 gives same std as 0-1 rand(), if want to use normal dist.
                x = x.div(quant_levels - 1).mul(diff).add(min_val)
            
            if lut_min is not None:
                pos_x = torch.relu(x)
                neg_x = x - pos_x
                lms = lut_min * max_val
                pos_x[pos_x < lms] = lms
                lms = lut_min * torch.abs(min_val)
                neg_x[neg_x > -lms] = -lms
                x = pos_x + neg_x

            return x

    @staticmethod
    def backward(ctx, grad_output):
        # STE
        return grad_output, None, None, None, None, None

class QuantizeStats(nn.Module):
    def __init__(self, percentile, use_clipping=True):
        super(QuantizeStats, self).__init__()
        self.register_buffer('running_min', torch.tensor(0.0))
        self.register_buffer('running_max', torch.tensor(0.0))
        self.max_calibration_steps = 1
        self.initial_calibration_steps = 0
        #self.register_buffer('calibration_done', torch.tensor(False))
        self.calibration_done = torch.tensor(False)
        self.activations = []
        self.percentile = percentile
        self.use_clipping = use_clipping

    def update(self, tensor):
        if self.use_clipping:
            if not self.calibration_done.item():
                self.initial_calibration_steps += 1
                finished = False

                if self.initial_calibration_steps >= self.max_calibration_steps:
                    finished = True
                    self.calibration_done = torch.tensor(True)

                with torch.no_grad():
                    self.activations.extend(tensor.detach().cpu().tolist())

                    if finished:
                        maximum = numpy.percentile(self.activations, self.percentile)
                        self.running_max = torch.tensor(maximum, device=tensor.device, dtype=tensor.dtype)
                        minimum = tensor.min()
                        minimum = minimum if minimum >= 0.0 else -maximum
                        self.running_min = torch.tensor(minimum, device=tensor.device, dtype=tensor.dtype)
                        self.activations.clear() # free the memory
                    else:
                        self.running_min = tensor.min()
                        self.running_max = tensor.max()
        
        else:
            alpha = 0.999
            with torch.no_grad():
                cur_min = tensor.min()
                cur_max = tensor.max()

                if self.initial_calibration_steps == 0:
                    self.initial_calibration_steps += 1
                    self.running_min = cur_min
                    self.running_max = cur_max
                else:
                    self.running_min = alpha * self.running_min + (1.0 - alpha) * cur_min
                    self.running_max = alpha * self.running_max + (1.0 - alpha) * cur_max



    def get(self):
        return self.running_min, self.running_max

class AllPassMORRCirculantMatMals(ONNBaseLayer):
    """
    All-pass MORR Linear layer, assumes (1) block-circulant matrix (2) differential rails (3) learnable balancing factors.
    J. Gu, et al., "SqueezeLight: Towards Scalable Optical Neural Networks with Multi-Operand Ring Resonators"
    https://doi.org/10.23919/DATE51398.2021.9474147
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    miniblock: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config=None,
        device: Device = torch.device("cpu"),
    ) -> None:
        super(AllPassMORRCirculantMatMals, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        miniblock_size = config.get("miniblock", 4)
        self.miniblock = miniblock_size
        self.grid_dim_x = int(np.ceil(self.in_features / miniblock_size))
        self.grid_dim_y = int(np.ceil(self.out_features / miniblock_size))
        self.in_features_pad = self.grid_dim_x * miniblock_size
        self.out_features_pad = self.grid_dim_y * miniblock_size

        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = 32
        self.in_bit = 32

        morr_config = config.get("MORRConfig", MORRConfig_20um_MQ)
        morr_init_val = config.get("morr_init", MORRConfig_20um_MQ)
        self.MORRConfig = morr_config
        self.morr_init = morr_init_val
        self.mrr_a = morr_config.attenuation_factor
        self.mrr_r = morr_config.coupling_factor
        self.trainable_morr_bias = config.get("trainable_morr_bias", MORRConfig_20um_MQ)
        self.trainable_morr_scale = config.get(
            "trainable_morr_scale", MORRConfig_20um_MQ
        )
        self.device = device
        ### calculate FWHM (rad)
        self.morr_fwhm = (
            -4
            * np.pi**2
            * morr_config.radius
            * morr_config.effective_index
            * (
                1 / morr_config.resonance_wavelength
                - 1 / (morr_config.resonance_wavelength - morr_config.bandwidth / 2)
            )
        )

        ### allocate parameters
        self.weight = None
        self.x_zero_pad = None
        self.morr_output_scale = None  ## learnable balancing factors implelemt by MRRs
        self.morr_input_bias = None  ## round-trip phase shift bias within MORR
        self.morr_input_scale = (
            None  ## scaling factor for the round-trip phase shift within MORR
        )
        self.morr_gain = (
            100 / (self.in_features // self.miniblock)
        ) ** 0.5  ## TIA gain, calculated such that output variance is around 1
        ### build trainable parameters
        self.build_parameters()

        ### quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, device=self.device)
        self.weight_quantizer = weight_quantize_fn(
            self.w_bit, alg="dorefa_pos"
        )  ## [0-1] positive only, maintain the original scale
        self.morr_output_scale_quantizer = weight_quantize_fn(
            self.w_bit, alg="dorefa_sym"
        )  ## [-1,1] full-range

        self.mrr_roundtrip_phase_to_tr = mrr_roundtrip_phase_to_tr_func(
            a=self.mrr_a, r=self.mrr_r, intensity=True
        )

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no gamma noise
        self.set_gamma_noise(0)
        ### default set no crosstalk
        self.disable_crosstalk()
        ### default set no phase variation
        self.disable_phase_variation()

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(morr_init=morr_init_val)
        self.finegrain_drop_mask = None

    def build_parameters(self) -> None:

        self.weight = Parameter(
            torch.ones(
                self.grid_dim_y,
                self.grid_dim_x,
                self.miniblock,
                device=self.device,
                dtype=torch.float,
            )
        )
        ### Learnable balancing factor (morr_output_scale)
        ### We use a single scaling factor for each block
        self.morr_output_scale = Parameter(
            torch.randn(1, 1, max(1, self.grid_dim_x // 2) + 1, 1, device=self.device)
        )
        if self.trainable_morr_bias:
            ### initialize with the finest-granularity, i.e., per mini-block
            self.morr_input_bias = Parameter(
                torch.zeros(
                    self.grid_dim_y,
                    self.grid_dim_x,
                    device=self.device,
                    dtype=torch.float,
                )
            )
        if self.trainable_morr_scale:
            ### initialize with the finest-granularity, i.e., per mini-block
            self.morr_input_scale = Parameter(
                torch.zeros(
                    self.grid_dim_y,
                    self.grid_dim_x,
                    device=self.device,
                    dtype=torch.float,
                )
            )

    def reset_parameters(self, morr_init: bool = False) -> None:
        ### nonlinear curve aware initialization
        if morr_init:
            ## initialize weight
            morr_uniform_(
                self.weight,
                MORRConfig=self.MORRConfig,
                n_op=self.miniblock,
                biased=self.w_bit >= 16,
                gain=2 if self.in_bit < 16 else 1,
            )  # quantization needs zero-center
            self.sigma_weight = self.weight.data.std().item()
            self.weight_quant_gain = None

            ## output distribution aware initialization to output scaling factor
            t1 = mrr_roundtrip_phase_to_tr_fused(
                torch.tensor([0]).float(), a=self.mrr_a, r=self.mrr_r, intensity=True
            )
            t2 = mrr_roundtrip_phase_to_tr_fused(
                torch.tensor([self.morr_fwhm * 2.4]).float(),
                a=self.mrr_a,
                r=self.mrr_r,
                intensity=True,
            )
            g = (
                (t2 - t1) / (2.4 * self.morr_fwhm)
            ).item()  ## 0~2.4 FWHM slope as a linear approximation

            self.sigma_out_scale = 4 / (3 * self.grid_dim_x**0.5 * g * self.morr_fwhm)
            self.out_scale_quant_gain = None
            init.normal_(self.morr_output_scale, 0, self.sigma_out_scale)
        else:
            init.kaiming_normal_(self.weight.data)
            init.kaiming_normal_(self.morr_output_scale.data)
            self.sigma_weight = self.weight.data.std().item()
            self.weight_quant_gain = None
            self.sigma_out_scale = self.morr_output_scale.data.std().item()
            self.out_scale_quant_gain = None

        if self.morr_input_bias is not None:
            self.morr_input_bias.data.zero_()
        if self.morr_input_scale is not None:
            ### after sigmoid, it cooresponds to 1 scale
            init.normal_(self.morr_input_scale.data, 2, 0.1)

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """

        raise NotImplementedError

    def build_weight(self, y: Tensor) -> Tensor:
        if self.w_bit < 16:
            ### differentiable quantizer based on STE to enable QAT (Dorefa-Net, arXiv 2016)
            weight = self.weight_quantizer(self.weight)

            ## rescale weights after quantization can maintain the initialization distribution
            if self.weight_quant_gain is None:
                self.weight_quant_gain = self.sigma_weight / weight.data.std()
            if self.trainable_morr_scale:
                morr_scale = self.morr_scale * self.weight_quant_gain
            else:
                morr_scale = self.weight_quant_gain
            weight = weight.mul(
                morr_scale
            )  ### gain factor from Tanh used in quantization

            ### quantize learnable balancing factor
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
        else:
            weight = self.weight.abs()  # positive only
            morr_output_scale = (
                self.morr_output_scale - self.morr_output_scale.data.mean()
            )

        if self.finegrain_drop_mask is not None:
            weight = weight.mul(self.finegrain_drop_mask.float())

        ## differential balancing factor concatenation
        scale = morr_output_scale[..., :-1, :]
        scale_pad = morr_output_scale[..., -1:, :]
        if self.grid_dim_x % 2 == 0:
            # even blocks
            scale = torch.cat([scale, -scale], dim=2)  # [1, 1, q, 1]
        else:
            # odd blocks
            if self.grid_dim_x > 1:
                scale = torch.cat([morr_output_scale, -scale], dim=2)  # [1, 1, q, 1]
            else:
                scale = scale_pad  # [1, 1, q, 1]
        morr_output_scale = scale.squeeze(-1).unsqueeze(0)  # [1 ,1, 1, q]

        return weight, morr_output_scale

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_gamma_noise(
        self, noise_std: float, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std

    def load_parameters(self, param_dict) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.weight_quantizer.set_bitwidth(w_bit)
        self.morr_output_scale_quantizer.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bitwidth(in_bit)

    def input_modulator(self, x: Tensor) -> Tensor:
        ### voltage to power, which is proportional to the phase shift
        return x * x

    def set_crosstalk_coupling_matrix(
        self, coupling_factor: float, drop_perc: float = 0
    ) -> None:
        ### crosstalk coupling matrix is a symmetric matrix, but the intra-MORR crosstalk can be taken as a round-trip phase shift scaling factor, which is proportional to the number of segments after pruned.
        ### drop-perc is the pruning percentage.
        assert 0 <= coupling_factor <= 1, logger.error(
            f"Coupling factor must in [0,1], but got {coupling_factor}"
        )

        self.crosstalk_factor = (
            1 + max(3, (self.miniblock * (1 - drop_perc) - 1)) * coupling_factor
        )

    def enable_crosstalk(self) -> None:
        self.enable_thermal_crosstalk = True

    def disable_crosstalk(self) -> None:
        self.enable_thermal_crosstalk = False

    def set_phase_variation(self, phase_noise_std: float = 0) -> None:
        self.phase_noise_std = phase_noise_std

    def enable_phase_variation(self) -> None:
        self.enable_phase_noise = True

    def disable_phase_variation(self) -> None:
        self.enable_phase_noise = False

    def enable_trainable_morr_scale(self) -> None:
        self.trainable_morr_scale = True

    def disable_trainable_morr_scale(self) -> None:
        self.trainable_morr_scale = False

    def enable_trainable_morr_bias(self) -> None:
        self.trainable_morr_bias = True

    def disable_trainable_morr_bias(self) -> None:
        self.trainable_morr_bias = False

    @property
    def morr_bias(self) -> Tensor:
        if self.morr_input_bias is None:
            return None
        # return 2 * self.morr_fwhm * torch.sigmoid(self.morr_input_bias.unsqueeze(0).unsqueeze(-1))
        return self.morr_fwhm * torch.tanh(
            self.morr_input_bias.unsqueeze(0).unsqueeze(-1)
        )

    @property
    def morr_scale(self) -> Tensor:
        if self.morr_input_scale is None:
            return None
        return torch.sigmoid(self.morr_input_scale.unsqueeze(-1)) + 0.2  # [p, q, 1]

    def propagate_morr(
        self, weight: Tensor, x: Tensor, morr_output_scale: Tensor
    ) -> Tensor:
        """
        @description: propagate through the analytically calculated transfer matrix of molg. We implement circulant matrix multiplication using fast circ matmul
        @param weight {torch.Tensor} two phase shifters in the MZI-based attenuators
        @param x {torch.Tensor} complex-valued input
        @param morr_output_scale {torch.Tensor} learnable balancing factors
        @return: y {torch.Tensor} output of attenuators
        """
        ### x : [bs, q, k]
        ### weights: [p, q, k]
        ### morr_output_scale: [1, 1, 1, q]

        ### input scaling [TCAD'21], must have valid ranges. too small will have dead neuron and not enough nonlinearity; too large will have larger power, cross-channel crosstalk. [0.2 - 1.2] will be suitable
        ## build circulant weight matrix
        # crosstalk on the weights are much cheaper to compute than on the phase shift
        if self.enable_thermal_crosstalk and self.crosstalk_factor > 1:
            weight = weight * self.crosstalk_factor
        weight = toeplitz(weight).unsqueeze(0)  # [1,  p, q, k, k]
        x = x.unsqueeze(1).unsqueeze(-1)  # [bs, 1, q, k, 1]
        x = weight.matmul(x).squeeze(-1)  # [bs, p, q, k]

        if self.enable_phase_noise and self.phase_noise_std > 1e-5:
            x = x + torch.zeros_like(x).normal_(0, self.phase_noise_std)

        ### input biasing [TCAD'21], must have valid ranges. too large will have power issue and cross-channel crosstalk. [-2FWHM ~ 0]
        if self.trainable_morr_bias:
            x = x - self.morr_bias

        ### Use theoretical transmission function for trainable MORR nonlinearity [TCAD'21]
        ### x is the phase detuning, x=0 means on-resonance
        ### phase: [bs, p, q, k]
        x = self.mrr_roundtrip_phase_to_tr(x)  # 3x faster than autograd

        ## implement balancing factor as dot-product
        """
        if(self.w_bit < 16):
            morr_output_scale = self.morr_output_scale_quantizer(self.morr_output_scale)
            if(self.sigma_out_scale_quant_gain is None):
                self.sigma_out_scale_quant_gain = self.sigma_out_scale / morr_output_scale.data.std().item()
            morr_output_scale = morr_output_scale.mul(self.sigma_out_scale_quant_gain)### gain factor from Tanh used in quantization
        else:
            morr_output_scale = self.morr_output_scale
        # morr_output_scale = morr_output_scale * self.morr_gain
        scale = morr_output_scale[..., :-1, :]
        scale_pad = morr_output_scale[..., -1:, :]

        # print("morr diff transmission:", end=", ")
        # diff = x[..., :x.size(2)//2,:]-x[..., x.size(2)//2:,:]
        # print_stat(diff)
        if(self.grid_dim_x % 2 == 0):
            #even blocks
            scale = torch.cat([scale, -scale], dim=2) # [1, 1, q, 1]
        else:
            # odd blocks
            if(self.grid_dim_x > 1):
                scale = torch.cat([morr_output_scale, -scale], dim=2) # [1, 1, q, 1]
            else:
                scale = scale_pad # [1, 1, q, 1]
        scale = scale.squeeze(-1).unsqueeze(0) # [1 ,1, 1, q]
        # print("output scale Q:", end=", ")
        # print_stat(scale[..., :scale.size(-1)//2])
        """
        x = morr_output_scale.matmul(x)  # [1, 1, 1, q] x [bs, p, q, k] = [bs, p, 1, k]
        x = x.flatten(1)  # [bs, p*k]
        return x

    def get_finegrain_drop_mask(self, topk: int) -> Tensor:
        if self.w_bit < 16:
            weight = self.weight_quantizer(self.weight.data)  # [p, q, k]
        else:
            weight = self.weight.data.abs()
        indices = weight.argsort(dim=-1)
        mask = torch.ones_like(weight, dtype=torch.bool, device=weight.device)

        drop_indices = indices[:, :, 0:-topk]
        mask.scatter_(2, drop_indices, 0)
        self.finegrain_drop_mask = mask
        return mask

    def apply_finegrain_drop_mask(self, mask: Tensor) -> None:
        if self.w_bit < 16:
            self.weight.data.masked_fill_(~mask.view_as(self.weight.data), -1000)
        else:
            self.weight.data.masked_fill_(~mask.view_as(self.weight.data), 0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        
        # load y as weight:
        self.weight.data.copy_(y)

        # if used in transformer
        is_transformer = len(x.shape) == 3
        if is_transformer:
            B, N, D = x.shape
        
        assert (
            x.size(-1) == self.in_features
        ), f"[E] Input dimension does not match the weight size {self.out_features, self.in_features}, but got input size ({tuple(x.size())}))"
        if self.in_bit < 16:
            x = self.input_quantizer(x)

        weight, morr_output_scale = self.build_weight()
        if self.in_features_pad > self.in_features:
            if self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0):
                self.x_zero_pad = torch.zeros(
                    x.size(0),
                    self.in_features_pad - self.in_features,
                    device=x.device,
                    dtype=x.dtype,
                )
            x = torch.cat([x, self.x_zero_pad], dim=1)

        x = x.view(-1, self.grid_dim_x, self.miniblock)

        ### modulation
        ### x: [bs, q, k] -> [bs, q, k]
        x = self.input_modulator(x)

        ### propagate through morr array
        ### x: [bs, q, k] -> [bs, p*k]
        x = self.propagate_morr(weight, x, morr_output_scale)

        if self.out_features < self.out_features_pad:
            x = x[..., : self.out_features]
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)

        # adjust output shape if used in transformer
        if is_transformer:
            x = x.view(B, N, self.out_features)
        return x

class MORRMHA(nn.Module):
    def __init__(self, embed_dim, heads):
        super(MORRMHA, self).__init__()
        assert embed_dim % heads == 0
        self.n_heads = heads
        self.Wq = AllPassMORRCirculantLinear(embed_dim, embed_dim)
        self.Wk = AllPassMORRCirculantLinear(embed_dim, embed_dim)
        self.Wv = AllPassMORRCirculantLinear(embed_dim, embed_dim)
        self.qmm1 = AllPassMORRCirculantMatMals()
        self.dropout_wq = nn.Dropout(0.1)
        self.dropout_wk = nn.Dropout(0.1)
        self.dropout_wv = nn.Dropout(0.1)
        self.qmm2 = AllPassMORRCirculantMatMals()
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

class _MORRGPT(nn.Module):
    def __init__(self, features, heads, tokenizer, layers, max_length):
        super(_MORRGPT, self).__init__()
        vocab_size = len(tokenizer) + 8 - len(tokenizer) % 8 # pad vocab size to 8-multiple for tensor core acceleration
        assert vocab_size % 8 == 0
        self.pos_embedding = nn.Embedding(max_length, features)
        self.word_embedding = nn.Embedding(vocab_size, features, padding_idx = tokenizer.pad_token_id)
        self.embedding_dropout = nn.Dropout(0.1)
        self.decoders = nn.ModuleList([MORRDecoderLayer(features, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(features)
        self.output_head = nn.Linear(features, vocab_size)
        nn.init.normal_(self.word_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward_embedding(self, x):
        embedded = self.word_embedding(x)
        return embedded

    def forward_attn(self, x):
        mask = make_autoregressive_mask_for(x)
        pos = make_position_indices_for(x)
        pos_embed = self.embedding_dropout(self.pos_embedding(pos) + x)
        decoded = pos_embed
        for layer in self.decoders:
            decoded = layer(decoded, mask)
        
        out = self.norm(decoded)
        return out

    def forward(self, x):
        embedded = self.forward_embedding(x)
        decoded = self.forward_attn(embedded)
        out = self.output_head(decoded)
        return out


class MORRGPT(pl.LightningModule):
    def __init__(self, features, heads, layers=6, max_length=1024):
        super().__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        self.transformer = _MORRGPT(features, heads, self.tokenizer, layers, max_length)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.lr = 0.0005
        self.photon_target = 0
        self.training_steps = 100000
        self.extracting = False
        self.use_adam = True

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, x):
        return self.transformer(x)

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        preds = self(xs)
        features = preds.size(2)
        preds = preds.view(-1, features)
        ys = ys.view(-1)
        loss = self.loss(preds, ys)
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        preds = self(xs)
        features = preds.size(2)
        preds = preds.view(-1, features)
        ys = ys.view(-1)
        loss = self.loss(preds, ys)
        self.val_loss.update(loss)

    def validation_epoch_end(self, outputs):
        self.log('validation loss', self.val_loss)

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        preds = self(xs)
        features = preds.size(2)
        preds = preds.view(-1, features)
        ys = ys.view(-1)
        loss = self.loss(preds, ys)
        self.test_loss.update(loss)
        if self.extracting:
            raise ValueError("Extraction done, aborting")

    def test_epoch_end(self, outputs):
        self.log('test loss', self.test_loss)
        self.log('photon target', self.photon_target)

    def configure_optimizers(self):
        if self.use_adam:
            decay = set()
            no_decay = set()
            blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

            for mn, m in self.named_modules():
                for pn, p in m.named_parameters(recurse=False):
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if 'bias' in pn:
                        no_decay.add(fpn)
                    elif 'weight' in pn and not isinstance(m, blacklist_weight_modules):
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)

            param_dict = {pn: p for pn, p in self.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay

            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.02},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]

            optimizer = torch.optim.AdamW(optim_groups, lr=self.lr)
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2500, num_training_steps=self.training_steps)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'name': 'Cosine LR scheduler'
                }
            }
        else:
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=1e-5)
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2500, num_training_steps=self.training_steps)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'name': 'Cosine LR scheduler'
                }
            }
    
    def replace_output_head(self, module):
        self.transformer.output_head = module

    def enable_quantization(self):
        for m in self.transformer.modules():
            if isinstance(m, AllPassMORRCirculantLinear) or isinstance(m, AllPassMORRCirculantMatMals):
                m.enable_quantization()

    def set_photon_target(self, n_photons):
        self.photon_target = n_photons
        for m in self.transformer.modules():
            if isinstance(m, AllPassMORRCirculantLinear) or isinstance(m, AllPassMORRCirculantMatMals):
                m.set_photon_target(n_photons)

    def set_quantized_eval(self, value=True):
        for m in self.transformer.modules():
            if isinstance(m, AllPassMORRCirculantLinear) or isinstance(m, AllPassMORRCirculantMatMals):
                print("setting quantized eval")
                m.force_quantized_eval = value

    def save(self, fname):
        torch.save(self.transformer.state_dict(), fname)

    def load(self, fname):
        self.transformer.load_state_dict(torch.load(fname))

    def enable_extraction(self):
        lin1 = self.transformer.decoders[0].ff.layer2
        lin1.extract_simulated = True
        lin1.extract_name = 'first_linear'
        lin2 = self.transformer.decoders[-1].ff.layer2
        lin2.extract_simulated = True
        lin2.extract_name = 'last_linear'
        attn1 = self.transformer.decoders[0].attn.qmm1
        attn1.extract_simulated = True
        attn1.extract_name = 'first_attn'
        attn2 = self.transformer.decoders[-1].attn.qmm1
        attn2.extract_simulated = True
        attn2.extract_name = 'last_attn'
        self.extracting = True
    