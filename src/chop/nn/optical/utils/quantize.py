# """
# Description:
# Author: Jiaqi Gu (jqgu@utexas.edu)
# Date: 2021-06-06 03:15:00
# LastEditors: Jiaqi Gu (jqgu@utexas.edu)
# LastEditTime: 2021-06-06 03:15:00
# """

import numpy as np
import torch
import logging


__all__ = [
    # "uniform_quantize_cpu",
    # "pact_quantize",
    # "PACT_Act",
    # "uniform_quantize",
    # "uniform_quantize_new",
    # "ewgs_quantize",
    "input_quantize_fn",
    "weight_quantize_fn",
]


def uniform_quantize(k, gradient_clip=False):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2**k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input

    return qfn.apply


############ add observer and new quant based on range and zeropoint for activation
def uniform_quantize_new(k, gradient_clip=False):
    # """
    # Support uniform quantization with auto-adjusted input data range
    # args:
    #     k: bitwidth
    #     scale, zeropoint: obtained from observer
    # """

    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, scale, zero_point):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2**k - 1)
                # out = torch.round(input * n) / n
                # out = (torch.clamp(torch.round(input / scale + zero_point), 0, n) - zero_point) * scale
                out = (
                    input.div(scale)
                    .add_(zero_point)
                    .round_()
                    .clamp_(0, n)
                    .sub_(zero_point)
                    .mul_(scale)
                )
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input, None, None

    return qfn.apply


class input_quantize_fn(torch.nn.Module):
    def __init__(
        self, in_bit, alg="dorefa", device=torch.device("cuda:0"), quant_ratio=1.0
    ):
        # """Input quantizer with Quant_Noise supported
        # Args:
        #     in_bit (int): Input quantization bitwidth.
        #     device (Device, optional): torch Device. Defaults to torch.device("cuda:0").
        #     quant_ratio (float, optional): Quantization ratio. Defaults to 1.0.
        # """
        super(input_quantize_fn, self).__init__()
        assert 1 <= in_bit <= 32
        self.in_bit = in_bit
        self.alg = alg
        assert alg in {
            "dorefa",
            "normal",
        }, f"Only support (dorefa, normal), but got {alg}"
        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logging.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}"
        )
        self.device = device

        # define quant style
        # dorefa: clamp to 0-1
        # normal: obtain scale and zero_point via observer

        if self.alg == "dorefa":
            self.uniform_q = uniform_quantize(k=in_bit)
        elif self.alg == "normal":
            self.uniform_q = uniform_quantize_new(k=in_bit)
            self.scale = None
            self.zero_point = None
            ### select scale and zero-point using EMA: exponential moving averages
            # AT: MovingAverageMinMaxObserver only support self-defined quant bitwidths for pytorch1.7
            # obs = torch.quantization.observer.MovingAverageMinMaxObserver(averaging_constant=0.01, dtype=torch.quint8,
            #     qscheme=torch.per_tensor_affine, reduce_range=False, quant_min=0, quant_max=2**self.in_bit-1)
            # Thus use our version
            ### torch version must be higher than 1.7
            if 1 <= self.in_bit <= 8:  # observer does not support higher than 8-bit
                self.obs = torch.quantization.observer.MovingAverageMinMaxObserver(
                    averaging_constant=0.01,
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_affine,
                    reduce_range=False,
                    quant_min=0,
                    quant_max=2**self.in_bit - 1,
                ).to(self.device)
            else:
                self.obs = None

    def set_bitwidth(self, bit: int) -> None:
        ### regenerate quantizer without changing observation statistics
        if bit != self.in_bit:
            if self.alg == "dorefa":
                self.uniform_q = uniform_quantize(k=bit)
            elif self.alg == "normal":
                self.uniform_q = uniform_quantize_new(k=bit)
        self.in_bit = bit

    def set_alg(self, alg: str) -> None:
        assert alg in {
            "dorefa",
            "normal",
        }, f"Only support (dorefa, normal), but got {alg}"
        if alg != self.alg:
            if alg == "dorefa":
                self.uniform_q = uniform_quantize(k=self.in_bit)
            elif alg == "normal":
                self.uniform_q = uniform_quantize_new(k=self.in_bit)
        self.alg = alg

    def set_quant_ratio(self, quant_ratio=None):
        if quant_ratio is None:
            ### get recommended value
            quant_ratio = [
                None,
                0.2,
                0.3,
                0.4,
                0.5,
                0.55,
                0.6,
                0.7,
                0.8,
                0.83,
                0.86,
                0.89,
                0.92,
                0.95,
                0.98,
                0.99,
                1,
            ][min(self.in_bit, 16)]
        assert 0 <= quant_ratio <= 1, logging.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}"
        )
        self.quant_ratio = quant_ratio

    def forward(self, x):
        if self.quant_ratio < 1 and self.training:
            ### implementation from fairseq
            ### must fully quantize during inference
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(
                1 - self.quant_ratio
            )
        else:
            quant_noise_mask = None

        if self.in_bit == 32:
            input_q = x
        elif self.in_bit == 1:
            x = x.clamp(0, 1)
            input_q = (self.uniform_q(x - 0.5) + 1) / 2
            if quant_noise_mask is not None:
                noise = input_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                ### unquantized inputs have to be clamped
                input_q = x + noise
        else:
            ### dorefa-style clamp for input data
            if self.alg == "dorefa":
                x = x.clamp(0, 1)
                input_q = self.uniform_q(x)
            elif self.alg == "normal":
                if self.obs is not None:
                    if self.training:
                        self.obs(x)
                    scale, zero_point = self.obs.calculate_qparams()
                    # convert scale and zero_point type from qint8
                    self.scale = scale.to(x)
                    self.zero_point = zero_point.to(x)
                    input_q = self.uniform_q(x, self.scale, self.zero_point)
                else:
                    input_q = x  # if no observer (in_bit > 8), do not quantize
            else:
                raise NotImplementedError

            # add noise
            if quant_noise_mask is not None:
                noise = input_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                ### unquantized inputs have to be clamped
                input_q = x + noise

        return input_q


class weight_quantize_fn(torch.nn.Module):
    def __init__(self, w_bit, mode="oconv", alg="dorefa", quant_ratio=1.0):
        # """Differentiable weight quantizer. Support different algorithms. Support Quant-Noise with partial quantization.

        # Args:
        #     w_bit (int): quantization bitwidth
        #     mode (str, optional): Different mode indicates different NN architectures. Defaults to "oconv".
        #     alg (str, optional): Quantization algorithms. [dorefa, dorefa_sym, qnn, dorefa_pos] Defaults to "dorefa".
        #     quant_ratio (float, optional): Quantization ratio to support full-precision gradient flow. Defaults to 1.0.
        # """
        super(weight_quantize_fn, self).__init__()
        assert 1 <= w_bit <= 32, logging.error(
            f"Only support 1 - 32 bit quantization, but got {w_bit}"
        )
        self.w_bit = w_bit
        self.alg = alg
        self.mode = mode
        assert alg in {"dorefa", "dorefa_sym", "qnn", "dorefa_pos"}, logging.error(
            f"Only support (dorefa, dorefa_sym, qnn, dorefa_pos) algorithms, but got {alg}"
        )
        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logging.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}"
        )
        self.uniform_q = uniform_quantize(k=w_bit, gradient_clip=True)

    def set_quant_ratio(self, quant_ratio=None):
        if quant_ratio is None:
            ### get recommended value
            quant_ratio = [
                None,
                0.2,
                0.3,
                0.4,
                0.5,
                0.55,
                0.6,
                0.7,
                0.8,
                0.83,
                0.86,
                0.89,
                0.92,
                0.95,
                0.98,
                0.99,
                1,
            ][min(self.w_bit, 16)]
        assert 0 <= quant_ratio <= 1, logging.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}"
        )
        self.quant_ratio = quant_ratio

    def set_bitwidth(self, bit: int) -> None:
        ### regenerate quantizer without changing observation statistics
        if bit != self.w_bit:
            self.uniform_q = uniform_quantize(k=bit, gradient_clip=True)
            self.w_bit = bit

    def forward(self, x):
        if self.quant_ratio < 1 and self.training:
            ### implementation from fairseq
            ### must fully quantize during inference
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(
                1 - self.quant_ratio
            )
        else:
            quant_noise_mask = None

        if self.w_bit == 32:
            weight_q = torch.tanh(x)
            weight_q = weight_q / torch.max(torch.abs(weight_q))
        elif self.w_bit == 1:
            if self.mode == "ringonn":
                weight_q = (self.uniform_q(x) / 4) + 0.5
            else:
                if self.alg == "dorefa":
                    E = x.data.abs().mean()
                    weight_q = (self.uniform_q(x / E) * E + E) / 2  # [0, E]
                    if quant_noise_mask is not None:
                        x = (x + E) / 2
                        noise = weight_q.data.sub_(x.data).masked_fill_(
                            quant_noise_mask, 0
                        )
                        ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                        weight_q = x + noise
                elif self.alg == "dorefa_sym":
                    E = x.data.abs().mean()
                    weight_q = self.uniform_q(x / E) * E  # [-E, E]
                    if quant_noise_mask is not None:
                        noise = weight_q.data.sub_(x.data).masked_fill_(
                            quant_noise_mask, 0
                        )
                        ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                        weight_q = x + noise
                else:
                    assert NotImplementedError
        else:
            if self.alg == "dorefa":
                weight = torch.tanh(x)  # [-1, 1]
                weight = weight / 2 / torch.max(torch.abs(weight.data)) + 0.5
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight)
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(
                        quant_noise_mask, 0
                    )
                    ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                    weight_q = weight + noise

            elif self.alg == "dorefa_sym":
                weight = torch.tanh(x)  # [-1, 1]
                r = torch.max(torch.abs(weight.data))
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight / (2 * r) + 0.5) * (2 * r) - r
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(
                        quant_noise_mask, 0
                    )
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = weight + noise
            elif self.alg == "dorefa_pos":
                weight = torch.tanh(x)  # [-1, 1]
                r = torch.max(torch.abs(weight.data))
                weight = weight + r
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight / (2 * r)) * 2 * r
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(
                        quant_noise_mask, 0
                    )
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = weight + noise

            elif self.alg == "qnn":
                x_min = torch.min(x.data)
                x_max = torch.max(x.data)
                x_range = x_max - x_min
                weight_q = self.uniform_q((x - x_min) / x_range) * x_range + x_min
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = x + noise
            else:
                assert NotImplementedError

        return weight_q
