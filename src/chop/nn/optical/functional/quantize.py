"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 03:15:00
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 03:15:00
"""

import numpy as np
import torch
import logging


__all__ = [
    "uniform_quantize_cpu",
    "pact_quantize",
    "PACT_Act",
    "uniform_quantize",
    "uniform_quantize_new",
    "ewgs_quantize",
    "input_quantize_fn",
    "weight_quantize_fn",
]


class uniform_quantize_cpu(object):
    def __init__(self, bits):
        super(uniform_quantize_cpu).__init__()
        self.bits = bits

    def __call__(self, input):
        if self.bits == 32:
            out = input
        elif self.bits == 1:
            out = np.sign(input)
        else:
            n = float(2**self.bits - 1)
            out = np.round(input * n) / n
        return out


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
    """
    Support uniform quantization with auto-adjusted input data range
    args:
        k: bitwidth
        scale, zeropoint: obtained from observer
    """

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
                out = input.div(scale).add_(zero_point).round_().clamp_(0, n).sub_(zero_point).mul_(scale)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input, None, None

    return qfn.apply


def ewgs_quantize(num_levels, gradient_clip=False, scaling_factor: float = 1e-3):
    class EWGS_quantizer(torch.autograd.Function):
        """
        Network Quantization with Element-wise Gradient Scaling, CVPR 2021
        https://github.com/cvlab-yonsei/EWGS/blob/main/CIFAR10/custom_modules.py
        x_in: continuous inputs within the range of [0,1]
        num_levels: number of discrete levels
        scaling_factor: backward scaling factor, typically fixed to 1e-3
        x_out: discretized version of x_in within the range of [0,1]
        """

        @staticmethod
        def forward(ctx, input):
            out = input.mul(num_levels - 1).round_().mul_(1/(num_levels - 1))

            ctx._scaling_factor = scaling_factor
            ctx.save_for_backward(input - out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            diff = ctx.saved_tensors[0]
            delta = ctx._scaling_factor
            scale = diff.mul_(grad_output.sign()).mul_(delta).add_(1)
            grad_input = grad_output * scale
            if gradient_clip:
                grad_input.clamp_(-1, 1)
            return grad_input

    return EWGS_quantizer.apply


class input_quantize_fn(torch.nn.Module):
    def __init__(self, in_bit, alg="dorefa", device=torch.device("cuda:0"), quant_ratio=1.0):
        """Input quantizer with Quant_Noise supported
        Args:
            in_bit (int): Input quantization bitwidth.
            device (Device, optional): torch Device. Defaults to torch.device("cuda:0").
            quant_ratio (float, optional): Quantization ratio. Defaults to 1.0.
        """
        super(input_quantize_fn, self).__init__()
        assert 1 <= in_bit <= 32
        self.in_bit = in_bit
        self.alg = alg
        assert alg in {"dorefa", "normal"}, f"Only support (dorefa, normal), but got {alg}"
        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logging.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
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
        assert alg in {"dorefa", "normal"}, f"Only support (dorefa, normal), but got {alg}"
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
        assert 0 <= quant_ratio <= 1, logging.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
        self.quant_ratio = quant_ratio

    def forward(self, x):
        if self.quant_ratio < 1 and self.training:
            ### implementation from fairseq
            ### must fully quantize during inference
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(1 - self.quant_ratio)
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
        """Differentiable weight quantizer. Support different algorithms. Support Quant-Noise with partial quantization.

        Args:
            w_bit (int): quantization bitwidth
            mode (str, optional): Different mode indicates different NN architectures. Defaults to "oconv".
            alg (str, optional): Quantization algorithms. [dorefa, dorefa_sym, qnn, dorefa_pos] Defaults to "dorefa".
            quant_ratio (float, optional): Quantization ratio to support full-precision gradient flow. Defaults to 1.0.
        """
        super(weight_quantize_fn, self).__init__()
        assert 1 <= w_bit <= 32, logging.error(f"Only support 1 - 32 bit quantization, but got {w_bit}")
        self.w_bit = w_bit
        self.alg = alg
        self.mode = mode
        assert alg in {"dorefa", "dorefa_sym", "qnn", "dorefa_pos"}, logging.error(
            f"Only support (dorefa, dorefa_sym, qnn, dorefa_pos) algorithms, but got {alg}"
        )
        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logging.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
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
        assert 0 <= quant_ratio <= 1, logging.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
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
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(1 - self.quant_ratio)
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
                        noise = weight_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                        ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                        weight_q = x + noise
                elif self.alg == "dorefa_sym":
                    E = x.data.abs().mean()
                    weight_q = self.uniform_q(x / E) * E  # [-E, E]
                    if quant_noise_mask is not None:
                        noise = weight_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
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
                    noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
                    ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                    weight_q = weight + noise

            elif self.alg == "dorefa_sym":
                weight = torch.tanh(x)  # [-1, 1]
                r = torch.max(torch.abs(weight.data))
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight / (2 * r) + 0.5) * (2 * r) - r
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = weight + noise
            elif self.alg == "dorefa_pos":
                weight = torch.tanh(x)  # [-1, 1]
                r = torch.max(torch.abs(weight.data))
                weight = weight + r
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight / (2 * r)) * 2 * r
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
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


# PACT activation: https://arxiv.org/pdf/1805.06085.pdf
class PACT_QuantFunc(torch.autograd.Function):
    r"""PACT (PArametrized Clipping acTivation) quantization function for activations.
        Implements a :py:class:`torch.autograd.Function` for quantizing activations in :math:`Q` bits using the PACT strategy.
        In forward propagation, the function is defined as

        .. math::
            \mathbf{y} = f(\mathbf{x}) = 1/\varepsilon \cdot \left\lfloor\mathrm{clip}_{ [0,\alpha) } (\mathbf{x})\right\rfloor \cdot \varepsilon

        where :math:`\varepsilon` is the quantization precision:

        .. math::
            \varepsilon = \alpha / (2^Q - 1)

        In backward propagation, using the Straight-Through Estimator, the gradient of the function is defined as

        .. math::
            \mathbf{\nabla}_\mathbf{x} \mathcal{L} &\doteq \mathbf{\nabla}_\mathbf{y} \mathcal{L}

        It can be applied by using its static `.apply` method:

    :param input: the tensor containing :math:`x`, the activations to be quantized.
    :type  input: `torch.Tensor`
    :param eps: the precomputed value of :math:`\varepsilon`.
    :type  eps: `torch.Tensor` or float
    :param alpha: the value of :math:`\alpha`.
    :type  alpha: `torch.Tensor` or float
    :param delta: constant to sum to `eps` for numerical stability (default unused, 0 ).
    :type  delta: `torch.Tensor` or float

    :return: The quantized input activations tensor.
    :rtype:  `torch.Tensor`
    """

    @staticmethod
    def forward(ctx, input, eps, alpha):
        where_input_clipped = (input < 0) | (input >= alpha)
        where_input_ltalpha = input < alpha
        ctx.save_for_backward(where_input_clipped, where_input_ltalpha)
        return ((input / (eps)).floor() * eps).clamp(0.0, alpha.data[0] - eps.data[0])

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        where_input_clipped, where_input_ltalpha = ctx.saved_tensors
        # zero = torch.zeros(1, device=where_input_nonclipped.device)
        grad_input = grad_output.masked_fill(where_input_clipped, 0)
        # grad_input = torch.where(where_input_nonclipped, grad_output, zero)
        grad_alpha = grad_output.masked_fill(where_input_ltalpha, 0).sum().expand(1)
        # grad_alpha = torch.where(where_input_gtalpha, grad_output, zero).sum().expand(1)
        return grad_input, None, grad_alpha


pact_quantize = PACT_QuantFunc.apply


class PACT_Act(torch.nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation.
    Implements a :py:class:`torch.nn.Module` to implement PACT-style activations. It is meant to replace :py:class:`torch.nn.ReLU`, :py:class:`torch.nn.ReLU6` and
    similar activations in a PACT-quantized network.
    This layer can also operate in a special mode, defined by the `statistics_only` member, in which the layer runs in
    forward-prop without quantization, collecting statistics on the activations that can then be
    used to reset the value of :math:`\alpha`.
    In this mode, the layer collects:
    - tensor-wise maximum value ever seen
    - running average with momentum 0.9
    - running variance with momentum 0.9
    """

    def __init__(
        self,
        precision=None,
        alpha=1.0,
        backprop_alpha=True,
        statistics_only=False,
        leaky=None,
        device=torch.device("cuda"),
    ):
        r"""Constructor. Initializes a :py:class:`torch.nn.Parameter` for :math:`\alpha` and sets
            up the initial value of the `statistics_only` member.
        :param precision: instance defining the current quantization level (default `None`).
        :type  precision: :py:class:`nemo.precision.Precision`
        :param alpha: the value of :math:`\alpha`.
        :type  alpha: `torch.Tensor` or float
        :param backprop_alpha: default `True`; if `False`, do not update the value of `\alpha` with backpropagation.
        :type  backprop_alpha: bool
        :param statistics_only: initialization value of `statistics_only` member.
        :type  statistics_only: bool
        """

        super(PACT_Act, self).__init__()
        self.precision = precision
        self.device = device
        self.alpha = torch.nn.Parameter(torch.Tensor((alpha,)).to(device), requires_grad=backprop_alpha)
        self.alpha_p = alpha
        self.statistics_only = statistics_only
        self.deployment = False
        self.eps_in = None
        self.leaky = leaky
        # self.requantization_factor = requantization_factor

        # these are only used to gather statistics
        self.max = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.min = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.running_mean = torch.nn.Parameter(torch.zeros_like(self.alpha.data).to(device), requires_grad=False)
        self.running_var = torch.nn.Parameter(torch.ones_like(self.alpha.data).to(device), requires_grad=False)

        self.precise = False

    def set_static_precision(self, limit_at_32_bits=True, **kwargs):
        r"""Sets static parameters used only for deployment."""
        # item() --> conversion to float
        # apparently causes a slight, but not invisibile, numerical divergence
        # between FQ and QD stages
        self.eps_static = self.alpha.clone().detach() / (2.0 ** (self.precision) - 1)
        self.alpha_static = self.alpha.clone().detach()
        # D is selected as a power-of-two
        D = 2.0 ** torch.ceil(torch.log2(self.requantization_factor * self.eps_static / self.eps_in))
        if not limit_at_32_bits:
            self.D = D
        else:
            self.D = min(D, 2.0 ** (32 - 1 - (self.precision)))

    def get_output_eps(self, eps_in):
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.
        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`
        """

        return self.alpha / (2.0 ** (self.precision) - 1)

    def reset_alpha(self, use_max=True, nb_std=5.0):
        r"""Reset the value of :math:`\alpha`. If `use_max` is `True`, then the highest tensor-wise value collected
            in the statistics collection phase is used. If `False`, the collected standard deviation multiplied by
            `nb_std` is used as a parameter
        :param use_max: if True, use the tensor-wise maximum value collected in the statistics run as new :math:`\alpha` (default True).
        :type  use_max: bool
        :param nb_std: number of standard deviations to be used to initialize :math:`\alpha` if `use_max` is False.
        :type  nb_std: float
        """

        if use_max:
            self.alpha.data[0] = self.max.item()
        else:
            self.alpha.data[0] = nb_std * torch.sqrt(self.running_var).item()

    def get_statistics(self):
        r"""Returns the statistics collected up to now.

        :return: The collected statistics (maximum, running average, running variance).
        :rtype:  tuple of floats
        """
        return self.max.item(), self.running_mean.item(), self.running_var.item()

    def forward(self, x):
        r"""Forward-prop function for PACT-quantized activations.

        See :py:class:`nemo.quant.pact_quant.PACT_QuantFunc` for details on the normal operation performed by this layer.
        In statistics mode, it uses a normal ReLU and collects statistics in the background.
        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`

        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`
        """

        if self.statistics_only:
            if self.leaky is None:
                x = torch.nn.functional.relu(x)
            else:
                x = torch.nn.functional.leaky_relu(x, self.leaky)
            with torch.no_grad():
                self.max[:] = max(self.max.item(), x.max())
                self.min[:] = min(self.min.item(), x.min())
                self.running_mean[:] = 0.9 * self.running_mean.item() + 0.1 * x.mean()
                self.running_var[:] = 0.9 * self.running_var.item() + 0.1 * x.std() * x.std()
            return x
        else:
            eps = self.alpha / (2.0 ** (self.precision) - 1)
            return pact_quantize(x, eps, self.alpha + eps)
