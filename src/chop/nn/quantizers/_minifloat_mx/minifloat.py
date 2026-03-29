"""
Minifloat quantize-dequantize simulation.
"""

import torch
from torch import Tensor

from .meta import MinifloatMeta, MinifloatTensorMeta
from .fake import extract_minifloat_component, compose_minifloat_component


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
    ori_dtype = tensor.dtype
    element = extract_minifloat_component(tensor, minifloat_meta)

    return compose_minifloat_component(
        element, minifloat_meta, output_dtype=output_dtype or ori_dtype
    )
