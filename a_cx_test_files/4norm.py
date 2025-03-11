from chop.nn.quantized import ViTAttentionInteger
import logging

import torch.nn as nn
import torch

from chop.nn.quantizers.integer import (
    integer_floor_quantizer,
)

logger = logging.getLogger("norm.models")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

def quantize(x, width, frac_width, by_pass=False):
    if not by_pass:
        x = integer_floor_quantizer(x, width, frac_width)
    return x
def get_dim_and_prodofdim(x, normalized_shape):
    dim = tuple(range(0 - len(normalized_shape), 0))
    num_vals = 1
    for items in dim:
        num_vals *= x.shape[items]
    return dim, num_vals
def isqrt(x:torch.Tensor):
    x = x.sqrt()
    x = x.reciprocal()
    return x
def _fixed_group_norm_2d_model(
    x: torch.Tensor,
    normalized_shape: tuple,
    q_config,
):
    #TODO: add hardware debug info
    logger.debug(f"Input: \n {x[0]}")
    dim, num_vals = get_dim_and_prodofdim(x, normalized_shape)
        
    # Mean calculation
    mu = x.mean(dim, keepdim=True)
    logger.debug(f"Mu: \n {mu[0]}")
    mu = quantize(mu, q_config["in_width"], q_config["in_frac_width"], q_config["by_pass"])
    logger.debug(f"Mu Quantized: \n {mu[0]}")

    # Variance calculation
    diff = x - mu
    logger.debug(f"Diff: \n {diff[0]}")

    squares = diff**2
    logger.debug(f"Squares: {squares[0]}")

    sum_squares = torch.sum(squares, dim, keepdim=True)

    sum_squares = quantize(sum_squares, q_config["variance_width"], q_config["variance_frac_width"], q_config["by_pass"])

    logger.debug("Num Values: %d" % (num_vals))
    var = sum_squares / num_vals
    var = quantize(var, q_config["variance_width"], q_config["variance_frac_width"], q_config["by_pass"])
    logger.debug(f"Variance: \n {var[0]}")

    inv_sqrt = isqrt(var + 1e-05)
    inv_sqrt = quantize(inv_sqrt, q_config["isqrt_width"], q_config["isqrt_frac_width"], q_config["by_pass"])
    logger.debug(f"INV SQRT INT: \n {inv_sqrt[0]}")

    # Norm calculation
    norm_out = diff * inv_sqrt
    logger.debug("Norm:")
    logger.debug(norm_out[0])

    norm_out = quantize(norm_out, q_config["out_width"], q_config["out_frac_width"], q_config["by_pass"])
    logger.debug(f"Norm (Casted): \n {norm_out[0]}")

    return norm_out 

if __name__ == "__main__":
    dim = 4
    head = 2

    torch.manual_seed(0)
    q_config = {
        "by_pass": False,
        "in_width":8,
        "in_frac_width":7,
        "variance_width":16,
        "variance_frac_width":8,
        "isqrt_width":16,
        "isqrt_frac_width":8,
        "out_width":8,
        "out_frac_width":4,
    }
    logger.setLevel(logging.DEBUG)
    x = torch.rand(1, dim)
    _x = _fixed_group_norm_2d_model(
    x, (4,), q_config)
    module = torch.nn.LayerNorm(dim,elementwise_affine=False, bias=False)
    print(_x)
    print(module(x))