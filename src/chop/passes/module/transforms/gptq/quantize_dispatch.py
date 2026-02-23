"""
Dispatch quantization to mxfp_quantizer_sim or mxint_quantizer_sim
based on format string, using Mase-style config dicts.
"""

import torch
from torch import Tensor

from chop.nn.quantizers.mxfp.mxfp import mxfp_quantizer_sim
from chop.nn.quantizers.mxfp.meta import MXFPMeta
from chop.nn.quantizers.mxint.mxint import mxint_quantizer_sim
from chop.nn.quantizers.mxint.meta import MXIntMeta


def _build_mxfp_meta(config: dict) -> MXFPMeta:
    return MXFPMeta(
        block_size=config["weight_block_size"],
        scale_exp_bits=8,
        element_exp_bits=config["weight_exponent_width"],
        element_frac_bits=config["weight_frac_width"],
        element_is_finite=True,
        round_mode="rn",
    )


def _build_mxint_meta(config: dict) -> MXIntMeta:
    return MXIntMeta(
        block_size=config["weight_block_size"],
        scale_bits=8,
        element_bits=config["weight_width"],
    )


def quantize_tensor(
    input: Tensor,
    block_dim: int,
    fmt: str,
    config: dict,
    quantile_search: bool,
    act_tensor: Tensor | None = None,
    dtype: torch.dtype | None = None,
    cali_batch_size: int = 32,
) -> Tensor:
    """
    Quantize a tensor using Mase quantizers, dispatching by format string.

    Args:
        input: Weight tensor to quantize.
        block_dim: Dimension for block quantization.
        fmt: "mxfp" or "mxint".
        config: Mase-style weight config dict with keys like
                weight_block_size, weight_exponent_width, weight_frac_width (mxfp)
                or weight_block_size, weight_width (mxint).
        quantile_search: Enable percentile-based clipping search.
        act_tensor: Optional activation tensor for calibration.
        dtype: Output dtype.
        cali_batch_size: Batch size for calibration search.
    """
    if fmt == "mxfp":
        meta = _build_mxfp_meta(config)
        return mxfp_quantizer_sim(
            input,
            block_dim=block_dim,
            mxfp_meta=meta,
            act_tensor=act_tensor,
            dtype=dtype,
            quantile_search=quantile_search,
            cali_batch_size=cali_batch_size,
        )
    elif fmt == "mxint":
        meta = _build_mxint_meta(config)
        return mxint_quantizer_sim(
            input,
            block_dim=block_dim,
            mxint_meta=meta,
            act_tensor=act_tensor,
            dtype=dtype,
            quantile_search=quantile_search,
            cali_batch_size=cali_batch_size,
        )
    else:
        raise ValueError(f"Unsupported GPTQ format: {fmt}. Use 'mxfp' or 'mxint'.")
