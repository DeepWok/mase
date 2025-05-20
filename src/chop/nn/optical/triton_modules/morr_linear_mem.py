"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-04-18 14:19:57
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-04-18 16:21:37
"""

from typing import Optional
import logging

import numpy as np
import torch
import torch.fft
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device

from ..utils import MORRConfig_20um_MQ
from ..utils import mrr_roundtrip_phase_to_tr_func, mrr_roundtrip_phase_to_tr_fused
from ..utils import toeplitz
from ..utils import morr_uniform_
from ..utils import input_quantize_fn, weight_quantize_fn
from ..modules.base_layer import ONNBaseLayer
from .morr_linear_kernel_mem import morr_linear_fn_mem

logger = logging.getLogger(__name__)

__all__ = ["AllPassMORRCirculantLinear"]


class TritonMemMORRLinear(ONNBaseLayer):
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
        config={},
        device: Device = torch.device("cpu"),
    ) -> None:
        super(TritonMemMORRLinear, self).__init__()
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

    def build_weight(self) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        output, *_ = morr_linear_fn_mem(
           x,
            self.weight,
            morr_input_bias = self.morr_input_bias,
            morr_output_scale = self.morr_output_scale,
            bias = None,
            morr_input_scale = self.morr_input_scale,
            morr_bias = self.morr_bias.detach(),
            grid_dim_x = self.grid_dim_x,
            grid_dim_y = self.grid_dim_y,
            miniblock = self.miniblock,
            enable_thermal_crosstalk=self.enable_thermal_crosstalk,
            crosstalk_factor=None if not self.enable_thermal_crosstalk else self.crosstalk_factor,
            enable_phase_noise=self.enable_phase_noise,
            phase_noise_std=None if not self.enable_phase_noise else self.phase_noise_std,
            trainable_morr_bias=self.trainable_morr_bias,
            mrr_a=self.mrr_a,
            mrr_r=self.mrr_r,
            finegrain_drop_mask=None,
            in_features = self.in_features,
            in_features_pad = self.in_features_pad,
            out_features = self.out_features,
            out_features_pad = self.out_features_pad,
            in_bit = self.in_bit,
            w_bit = self.w_bit,
            morr_fwhm = self.morr_fwhm,
            sigma_weight=self.sigma_weight,
            trainable_morr_scale=self.trainable_morr_scale, # bool
            morr_scale=self.morr_scale,
            weight_quant_gain=self.weight_quant_gain,
            seed = 42,
        )
        return output