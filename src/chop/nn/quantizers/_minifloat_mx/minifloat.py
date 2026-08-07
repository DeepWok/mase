"""
Minifloat quantize-dequantize simulation.
"""

import torch
from torch import Tensor

from .meta import MinifloatMeta, MinifloatTensorMeta
from .fake import quantize_minifloat_value


def minifloat_quantizer_sim(
    tensor: Tensor,
    minifloat_meta: MinifloatMeta,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Quantize and dequantize a tensor using minifloat format.

    Args:
        tensor: Input tensor to quantize
        minifloat_meta: Minifloat format specification
        output_dtype: Desired output dtype (default: same as input)

    Returns:
        Dequantized tensor
    """
    value = quantize_minifloat_value(tensor, minifloat_meta)
    return value.to(output_dtype or tensor.dtype)
