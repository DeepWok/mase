"""Compatibility entry points for the legacy hardware-shaped MXINT API."""

import torch

from .mxint.mxint import mxint_quantizer


def mxint_quant_block(
    x, width: int = 12, exponent_width: int = 6, exponent: int = None
):
    """Quantize one block using the canonical MXINT implementation.

    ``exponent`` is retained for callers that supply an explicit integer
    quantum. Normal scale selection delegates to :func:`mxint_quantizer`, so
    rounding, signed range, E8M0 bias, and zero-block behavior cannot drift
    from the decode quantizer.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.numel() == 0:
        raise ValueError("MXINT blocks must contain at least one element")

    if exponent is None:
        return mxint_quantizer(
            x,
            block_size=x.numel(),
            element_bits=width,
            block_dim=-1,
            scale_bits=exponent_width,
        )

    scale_bias = 2 ** (exponent_width - 1) - 1
    scale_min = -scale_bias
    scale_max = 2**exponent_width - 1 - scale_bias
    exponent_value = int(exponent)
    if not scale_min <= exponent_value <= scale_max:
        raise ValueError(
            f"MXINT exponent {exponent_value} is outside "
            f"[{scale_min}, {scale_max}]"
        )
    magnitude_max = 2 ** (width - 1) - 1
    elements = (x / 2**exponent_value).round().clamp(
        min=-magnitude_max,
        max=magnitude_max,
    )
    return elements * 2**exponent_value


def mxint_hardware(tensor, q_config, parallelism):
    """Quantize blocks selected by the legacy two-dimensional API.

    The reshape only selects blocks. Arithmetic is delegated to canonical
    MXINT, preserving the input dtype, device, and autograd path.
    """
    original_shape = tensor.shape
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    if len(parallelism) == 1:
        parallelism = [1, parallelism[0]]

    p1 = parallelism[0]
    p0 = parallelism[1]
    t1 = tensor.shape[-2]
    t0 = tensor.shape[-1]
    if t1 % p1 != 0 or t0 % p0 != 0:
        raise ValueError(
            "MXINT parallelism must divide the final two tensor dimensions: "
            f"shape=({t1}, {t0}), parallelism=({p1}, {p0})"
        )
    reshaped_tensor = tensor.reshape(-1, t1 // p1, p1, t0 // p0, p0).permute(
        0, 1, 3, 2, 4
    )

    blocked_tensor = reshaped_tensor.reshape(-1, p1 * p0)
    exponent = q_config.get("exponent")
    if exponent is None:
        blocked_tensor = mxint_quantizer(
            blocked_tensor,
            block_size=p1 * p0,
            element_bits=q_config.get("width", 12),
            block_dim=-1,
            scale_bits=q_config.get("exponent_width", 6),
        )
    else:
        blocked_tensor = torch.stack(
            [mxint_quant_block(block, **q_config) for block in blocked_tensor]
        )
    qtensor = (
        blocked_tensor.reshape(-1, t1 // p1, t0 // p0, p1, p0)
        .permute(0, 1, 3, 2, 4)
        .reshape(original_shape)
    )
    return qtensor
