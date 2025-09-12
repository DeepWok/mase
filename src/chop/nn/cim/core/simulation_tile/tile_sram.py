
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .quantization import scale_integer_quantizer
from logging import getLogger
logger = getLogger(__name__)


def _runtime_rescale(
    x: Tensor, exponent_bits: int = 1, mantissa_bits: int = 1, rescale_dim: str = "element"
):
    """
    The rescaling in side the digital mm
    """

    # if rescale_dim == "element":
    #     max_exponent = torch.log2(x).ceil()
    # elif rescale_dim == "vector":
    #     max_exponent = (torch.abs(x) + 1e-8).max(dim=-1, keepdim=True).values.log2().ceil()
    # else:
    #     raise ValueError(f"Invalid rescale_dim: {rescale_dim}")
    
    max_exponent = (torch.abs(x) + 1e-8).max(dim=-1, keepdim=True).values.log2().ceil()
    exponent_min = -2**(exponent_bits - 1)
    exponent_max = 2**(exponent_bits - 1) - 1
    max_exponent = torch.clamp(max_exponent, exponent_min, exponent_max)

    mantissa = x / 2**max_exponent
    mantissa_max = 2**mantissa_bits - 1
    mantissa_min = -2**mantissa_bits

    # recast mantissa
    mantissa = torch.clamp((mantissa * 2**mantissa_bits).round(), mantissa_min, mantissa_max)
    mantissa = mantissa / 2**mantissa_bits

    return mantissa * (2**max_exponent)

def approximate_mode(x: Tensor, weight: Tensor):
    """
    The approximate mode is to simulate the lossy digital multiplication 
    and the lossy digital accumulation
    """
    output = x @ weight
    logger.info(f"The approximate mode is implemented with 5.3% noise")
    output = output + output * torch.randn_like(output) * 0.053 # 5.3% noise

    return output

def sram_tile(x: Tensor, weight: Tensor, config: dict):
    '''
    There is two mode to conducting the digital mm, 
    For the first accurate mode, 
    we only need to rescale the x and weight, 
    and then quantize output to simulate the behaviour

    For the second mode, 
    we need to simulate the lossy digital multiplication 
    and the lossy digital accumulation
    '''
    
    x_quant_type = config.get("x_quant_type")
    weight_quant_type = config.get("weight_quant_type")

    if x_quant_type == "e4m3":
        qx = _runtime_rescale(x, 4, 3, config.get("rescale_dim", "vector"))
    elif x_quant_type == "e5m2":
        qx = _runtime_rescale(x, 5, 2, config.get("rescale_dim", "vector"))
    elif x_quant_type == "e8m7":
        qx = _runtime_rescale(x, 8, 7, config.get("rescale_dim", "vector"))
    elif x_quant_type == "int4":
        qx = scale_integer_quantizer(x, 4, True, 1.0)
    elif x_quant_type == "int8":
        qx = scale_integer_quantizer(x, 8, True, 1.0)
    else:
        qx = x

    weight = weight.transpose(-1, -2) # the rescale dimension should be in the -2 dimension 

    if weight_quant_type == "e4m3":
        qweight = _runtime_rescale(weight, 4, 3, config.get("rescale_dim", "vector"))
    elif weight_quant_type == "e5m2":
        qweight = _runtime_rescale(weight, 5, 2, config.get("rescale_dim", "vector"))
    elif weight_quant_type == "e8m7":
        qweight = _runtime_rescale(weight, 8, 7, config.get("rescale_dim", "vector"))
    elif weight_quant_type == "int4":
        qweight = scale_integer_quantizer(weight, 4, True, 1.0)
    elif weight_quant_type == "int8":
        qweight = scale_integer_quantizer(weight, 8, True, 1.0)
    else:
        qweight = weight


    qweight = qweight.transpose(-1, -2) # permute back
    
    if config.get("approximate_mode", False):
        result = approximate_mode(qx, qweight)
    else:
        result = qx @ qweight # Considering in the flow of the paper there is no cast while sending back to AHB, so no cast in the end
    result = result

    return result

    