import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import scale_integer_quantizer

def reram_tile(x, weight, config):
    x = scale_integer_quantizer(
        x, 
        config.get("num_bits", 8), 
        True, 
        config.get("quantile", 1.0)
    )
    weight = weight + torch.randn_like(weight) * config.get("weight_noise", 0.0)
    return x @ weight
