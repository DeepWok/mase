from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from chop.nn.quantizers import (
    integer_quantizer, integer_floor_quantizer
)

from math import ceil, log2

def _int_layer_norm_2d_model(
    x: torch.Tensor,
    normalized_shape: tuple or int,
    weight = None,
    bias = None,
    eps = 1e-5,
    q_config = {},
):
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
        x = (x+eps).sqrt()
        x = x.reciprocal()
        return x
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    dim, num_vals = get_dim_and_prodofdim(x, normalized_shape)

    x = quantize(x, q_config.get("data_in_width"), q_config.get("data_in_frac_width"), q_config.get("by_pass"))
    num_vals_frac_width = ceil(log2(num_vals))
    inv_num_vals_quant = quantize(torch.tensor(1/num_vals),num_vals_frac_width + 2, num_vals_frac_width)
    # Mean calculation
    mu_acc = x.sum(dim, keepdim=True)
    mu = mu_acc * inv_num_vals_quant
    mu = quantize(mu, q_config.get("data_in_width"), q_config.get("data_in_frac_width"), q_config.get("by_pass"))
    #I hope the output precision here should be $clog2
    # Variance calculation
    diff = x - mu

    squares = diff**2

    sum_squares = torch.sum(squares, dim, keepdim=True)

    var = sum_squares * inv_num_vals_quant
    var = quantize(var, q_config.get("isqrt_in_width"), q_config.get("isqrt_in_frac_width"), q_config.get("by_pass"))

    inv_sqrt = isqrt(var)
    inv_sqrt = quantize(inv_sqrt, q_config.get("isqrt_out_width"), q_config.get("isqrt_out_frac_width"), q_config.get("by_pass"))

    # Norm calculation
    norm_out = diff * inv_sqrt

    norm_out = quantize(norm_out, q_config.get("data_out_width"), q_config.get("data_out_frac_width"), q_config.get("by_pass"))
    return norm_out 

class IntLayerNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, normalized_shape, weight, bias, eps, config):
        with torch.enable_grad():
            layernormed = nn.functional.layer_norm(
                input, normalized_shape, weight, bias, eps
            )
        ctx.save_for_backward(input, layernormed)
        output = _int_layer_norm_2d_model(
            input, normalized_shape, weight, bias, eps, config
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, layernormed = ctx.saved_tensors
        (grad_input,) = torch.autograd.grad(
            layernormed, input, grad_outputs=grad_output
        )
        return grad_input, None, None, None, None, None



class _LayerNormBase(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        assert elementwise_affine == False, "elementwise_affine not supported!"
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.bypass:
            x = self.x_quantizer(x)
        return F.layer_norm(x, self.normalized_shape, None, None)


class LayerNormInteger(_LayerNormBase):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        x_width, x_frac_width = config.get("data_in_width"), config.get("data_in_frac_width")
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )

class LayerNormIntegerFloor(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        assert elementwise_affine == False, "elementwise_affine not supported!"
        assert config is not None, "config is None!"
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.config = config
        self.bypass = config.get("bypass", False)

    def forward(self, x: Tensor) -> Tensor:
        return IntLayerNormFunc.apply(x, self.normalized_shape, self.weight, self.bias, self.eps, self.config)
