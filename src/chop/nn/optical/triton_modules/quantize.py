import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def uniform_quantize(x: tl.tensor, k, gradient_clip=False):
    if k == 32:
        out = input
    elif k == 1:
        out = tl.where(x >= 0, 1.0, -1.0)
    else:
        n = float(2 ** k - 1)
        out = tl.extra.cuda.libdevice.rint(x * n) / n

    return out


def uniform_quantize_new(x: tl.tensor, k, scale, zero_point, gradient_clip=False):
    if k == 32:
        out = x
    elif k == 1:
        out = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
    else:
        n = float(2 ** k - 1)
        out = tl.div(x, scale)
        out = out + zero_point
        out = tl.extra.cuda.libdevice.rint(out)
        out = tl.clamp(out, 0.0, n)
        out = out - zero_point
        out = out * scale
    return out


@triton.jit
def _input_quantize_fn(
    x: tl.tensor, quant_ratio, training, in_bit, alg,  # self.training
):
    # init
    if alg == "dorefa":
        uniform_q = uniform_quantize(k=in_bit)
    elif alg == "normal":
        uniform_q = uniform_quantize_new(k=in_bit)
        scale = None
        zero_point = None
    # TODO: fix for triton
    if 1 <= in_bit <= 8:  # observer does not support higher than 8-bit
        obs = torch.quantization.observer.MovingAverageMinMaxObserver(
            averaging_constant=0.01,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            quant_min=0,
            quant_max=2 ** in_bit - 1,
        )
    else:
        obs = None

    if quant_ratio > 1.0 and training:
        rand_vals = tl.random(x.shape)
        quant_noise_mask = tl.where(rand_vals > quant_ratio, 1, 0)
    else:
        quant_noise_mask = None

    if in_bit == 32:
        input_q = x
    elif in_bit == 1:
        x = tl.clamp(x, 0.0, 1.0)
        input_q = (uniform_q(x - 0.5) + 1) / 2
        if quant_noise_mask is not None:
            noise = input_q - x
            masked_noise = tl.where(quant_noise_mask, 0.0, noise)
            input_q = x + masked_noise
    else:
        ### dorefa-style clamp for input data
        if alg == "dorefa":
            x = tl.clamp(x, 0.0, 1.0)
            input_q = uniform_q(x)
        elif alg == "normal":
            if obs is not None:
                if training:
                    obs(x)
                scale, zero_point = obs.calculate_qparams()
                # convert scale and zero_point type from qint8
                scale = scale.to(x.dtype)
                zero_point = zero_point.to(x.dtype)
                input_q = uniform_q(x, scale, zero_point)
            else:
                input_q = x  # if no observer (in_bit > 8), do not quantize
        else:
            # raise NotImplementedError
            input_q = tl.zeros_like(x)
        # add noise
        if quant_noise_mask is not None:
            noise = input_q - x
            masked_noise = tl.where(quant_noise_mask, 0.0, noise)
            input_q = x + masked_noise

    return input_q


def _weight_quantize_fn(w: tl.tensor):
    pass
