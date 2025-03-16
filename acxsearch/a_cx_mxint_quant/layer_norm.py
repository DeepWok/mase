from torch import nn
import torch

from .quantizers import mxint_quant_block, mxint_hardware
from chop.nn.quantizers import integer_floor_quantizer, integer_quantizer
from torch import Tensor
from math import ceil, log2

def mxint_layer_norm(
    x: torch.Tensor,
    normalized_shape: tuple or int,
    weight=None,
    bias=None,
    eps=1e-5,
    q_config={},
):
    def quantize(x, width, frac_width, by_pass=False, floor=False):
        if not by_pass:
            if floor:   
                x = integer_floor_quantizer(x, width, frac_width)
            else:
                x = integer_quantizer(x, width, frac_width)
        return x

    def get_dim_and_prodofdim(x, normalized_shape):
        dim = tuple(range(0 - len(normalized_shape), 0))
        num_vals = 1
        for items in dim:
            num_vals *= x.shape[items]
        return dim, num_vals
    '''
        actually, we cannot assume that the input is quantized
    '''
    def isqrt(x: torch.Tensor):
        x = (x + eps).sqrt()
        x = x.reciprocal()
        return x

    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    dim, num_vals = get_dim_and_prodofdim(x, normalized_shape)
    inv_num_vals = torch.tensor(1 / num_vals)

    acc_out_width = ceil(log2(num_vals)) + q_config.get("data_in_width")
    inv_num_vals_quant_0 = 2**acc_out_width // num_vals / 2**acc_out_width
    # Mean calculation
    mu_acc = x.sum(dim, keepdim=True)
    mu = mu_acc * inv_num_vals_quant_0
    mu = quantize(
        mu,
        q_config.get("data_in_width"),
        q_config.get("data_in_frac_width"),
        q_config.get("by_pass"),
        True,
    )
    # I hope the output precision here should be $clog2
    # Variance calculation
    diff = x - mu

    squares = diff**2
    sum_squares = torch.sum(squares, dim, keepdim=True)
    squares_adder_tree_width = 2 * q_config.get("data_in_width") + ceil(log2(num_vals))
    inv_num_vals_quant_1 = 2**squares_adder_tree_width // num_vals / 2**squares_adder_tree_width
    var = sum_squares * inv_num_vals_quant_1
    var = quantize(
        var,
        squares_adder_tree_width + 2,
        2*q_config.get("data_in_width") - 2,
        floor=True,
    )
    var, mvar, evar = mxint_hardware(
        var,
        {
            "width": q_config.get("isqrt_in_width"),
            "exponent_width": 6,
        },
        parallelism=[1, 1],
    )
    
    var, mvar, evar = mxint_hardware(
        var,
        {
            "width": q_config.get("isqrt_in_width"),
            "exponent_width": 6,
        },
        parallelism=[1, 1],
    )
    mvar[evar %2 !=0] *= 2
    evar[evar %2 !=0] -= 1
    minv_sqrt = isqrt(mvar/2**(q_config.get("isqrt_in_width") - 1))
    minv_sqrt = integer_quantizer(minv_sqrt, q_config.get("isqrt_out_width"), q_config.get("isqrt_out_frac_width"))
    einv_sqrt = -evar/2

    inv_sqrt = minv_sqrt * 2**einv_sqrt

    # Norm calculation
    mnorm_out = diff * minv_sqrt
    enorm_out = einv_sqrt
    mnorm_out = quantize(
        mnorm_out,
        q_config.get("data_out_width"),
        q_config.get("data_out_frac_width"),
        q_config.get("by_pass"),
        floor=True,
    )
    qnorm_out = mnorm_out*2**einv_sqrt
    if weight is not None:
        qweight, mweight, eweight = mxint_hardware(weight, 
                                {
                                    "width": q_config.get("weight_width"),
                                    "exponent_width": q_config.get("weight_exponent_width"),
                                    "round_bits": 4
                                },
                                q_config.get("weight_parallelism"))
        qnorm_out = qnorm_out * qweight
        if bias is not None:
            qbias, mbias, ebias = mxint_hardware(bias, 
                                {
                                    "width": q_config.get("bias_width"),
                                    "exponent_width": q_config.get("bias_exponent_width"),
                                    "round_bits": 4
                                },
                                q_config.get("bias_parallelism"))
            qnorm_out = qnorm_out + qbias
    qnorm_out, mnorm_out, enorm_out = mxint_hardware(qnorm_out, 
                                {
                                    "width": q_config.get("data_out_width"),
                                    "exponent_width": q_config.get("data_out_exponent_width"),
                                    "round_bits": 4
                                },
                                q_config.get("data_out_parallelism"))
    return qnorm_out, mnorm_out, enorm_out

def layer_norm_hardware(
    x: torch.Tensor,
    normalized_shape: tuple or int,
    weight=None,
    bias=None,
    eps=1e-5,
    q_config=None,
):
    qx, mx, ex = mxint_quant_block(x, q_config["data_in_width"], q_config["data_in_exponent_width"])
    qnorm_out, _, _ = mxint_layer_norm(qx, normalized_shape, weight, bias, eps, q_config)
    return qnorm_out

class MXIntLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = False,
        bias: bool = False,
        q_config=None,
    ) -> None:
        self.q_config = q_config
        super().__init__(normalized_shape, eps, elementwise_affine, bias)

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm_hardware(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            q_config=self.q_config,
        )