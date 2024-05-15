import numpy
import torch


from .utils import (
    my_clamp,
    my_round,
    my_clamp,
    binarised_bipolar_op,
    binarised_zeroOne_op,
    binarised_zeroScaled_op,
    ternarised_scaled_op,
    ternarised_op,
)


def binary_quantizer(
    x: torch.Tensor | numpy.ndarray, stochastic: bool = False, bipolar: bool = False
):
    """
    - Do binary quantization to input
    - Optionally do stochastic quantization

    ---
    - forward:
    - backward: STE

    ---
    stochastic: enable stochastic quantization otherwise threshold is defaulted to 0
    positive_binarized: binarized input to {0, -1} if enabled else binarized input to {-1, 1}

    ---
    Refer to https://arxiv.org/pdf/1603.01025.pdf

    """

    if stochastic:
        x_sig = my_clamp((x + 1) / 2, 0, 1)
        x_rand = (
            torch.rand_like(x)
            if isinstance(x, torch.Tensor)
            else numpy.random.rand(*x.shape)
        )

        x = (
            binarised_bipolar_op(x_sig, x_rand)
            if bipolar
            else binarised_zeroScaled_op(x_sig, x_rand)
        )
    else:
        x = binarised_bipolar_op(x, 0) if bipolar else binarised_zeroScaled_op(x, 0)

    return x


def binarize(x):
    # Element-wise clipping
    clipped = torch.clamp(x, -1, 1)

    # Element-wise rounding to the closest integer with gradient propagation
    rounded = torch.sign(clipped)

    # Element-wise addition with gradient stop
    return clipped + (rounded - clipped).detach()


def residual_sign_quantizer(levels, x_quantizer, means, x):
    """
    - Do residual binarization quantization to input
    ---
    levels: number of residual levels
    x_quantizer: input quantizer to use for binarization (This is usually some binarizor)
    means: the pre-obtained residual batch
    ---
    Refer to https://arxiv.org/pdf/1603.01025.pdf
    """
    x_quantizer = binarize
    resid = x
    out_bin = 0

    if levels == 1:
        for l in range(levels):
            # out=binarize(resid)*K.abs(self.means[l])
            out = x_quantizer(resid) * torch.abs(means[l])
            # out_bin=out_bin+out
            out_bin = out_bin + out  # no gamma per level
            resid = resid - out
    elif levels == 2:
        out = x_quantizer(resid) * torch.abs(means[0])
        out_bin = out
        resid = resid - out
        out = x_quantizer(resid) * torch.abs(means[1])
        out_bin = torch.stack([out_bin, out], dim=0)
        resid = resid - out
    elif levels == 3:
        out = x_quantizer(resid) * torch.abs(means[0])
        out_bin1 = out
        resid = resid - out
        out = x_quantizer(resid) * torch.abs(means[1])
        out_bin2 = out
        resid = resid - out
        out = x_quantizer(resid) * torch.abs(means[2])
        out_bin3 = out
        resid = resid - out
        out_bin = torch.stack([out_bin1, out_bin2, out_bin3], dim=0)

    return out_bin
