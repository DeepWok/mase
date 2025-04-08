from torch import nn
import torch

from .quantizers import mxint_quant_block, mxint_hardware
from chop.nn.quantizers import integer_floor_quantizer, integer_quantizer
from torch import Tensor
from math import ceil, log2
from chop.tools import get_logger
logger = get_logger(__name__)

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

    qx, mx, ex = mxint_quant_block(
        x, 
        width=q_config.get("data_in_width"), 
        exponent_width=q_config.get("data_in_exponent_width"),
        round_bits=4,
    )
    qx = mx / 2**(q_config.get("data_in_width") - 1)

    norm_in_width = q_config.get("norm_in_int_width") + q_config.get("norm_in_frac_width")
    norm_in_frac_width = q_config.get("norm_in_frac_width")
    qx = quantize(qx, norm_in_width, norm_in_frac_width, floor=False)

    if q_config.get("enable_mean_sqrt_noise"):
        mu_acc = qx.sum(dim, keepdim=True)
        acc_out_width = ceil(log2(num_vals)) + norm_in_width
        # Mean calculation
        mu = mu_acc * quantize(inv_num_vals, acc_out_width + 2, acc_out_width)
        mu = quantize(
            mu,
            norm_in_width,
            norm_in_frac_width,
            floor=False,
        )
        # I hope the output precision here should be $clog2
        # Variance calculation
        diff = qx - mu
        squares = diff**2
        sum_squares = torch.sum(squares, dim, keepdim=True)
        squares_adder_tree_width = 2 * norm_in_width + ceil(log2(num_vals))
        
        var = sum_squares * quantize(inv_num_vals, squares_adder_tree_width + 2, squares_adder_tree_width)
        var_width = q_config.get("var_int_width") + q_config.get("var_frac_width")
        var_frac_width = q_config.get("var_frac_width")
        var = quantize(
            var,
            var_width,
            var_frac_width,
            floor=False,
        )
        if q_config.get("enable_mxint_var"):
            isqrt_in_width = q_config.get("isqrt_in_width")
            isqrt_out_width = q_config.get("isqrt_out_int_width") + q_config.get("isqrt_out_frac_width")
            isqrt_out_frac_width = q_config.get("isqrt_out_frac_width")
            var, mvar, evar = mxint_hardware(
                var,
                {
                    "width": isqrt_in_width,
                    "exponent_width": 8,
                },
                parallelism=[1, 1],
            )
            mvar[evar %2 !=0] *= 2
            evar[evar %2 !=0] -= 1
            minv_sqrt = isqrt(mvar/2**(isqrt_in_width - 1))
            minv_sqrt = integer_quantizer(minv_sqrt, isqrt_out_width, isqrt_out_frac_width)
            einv_sqrt = -evar/2

            inv_sqrt = minv_sqrt * 2**einv_sqrt
        else:
            inv_sqrt = isqrt(var)
        # Norm calculation
        norm_out = diff * inv_sqrt
    else:
        norm_out = nn.functional.layer_norm(qx, normalized_shape, None, None, eps)
    norm_out_width = q_config.get("norm_out_int_width") + q_config.get("norm_out_frac_width")
    norm_out_frac_width = q_config.get("norm_out_frac_width")
    norm_out = quantize(
        norm_out,
        norm_out_width,
        norm_out_frac_width,
        floor=False,
    )
    if weight is not None:
        norm_out = norm_out * weight
        if bias is not None:
            norm_out = norm_out + bias
    return norm_out

def layer_norm_hardware(
    x: torch.Tensor,
    normalized_shape: tuple or int,
    weight=None,
    bias=None,
    eps=1e-5,
    q_config=None,
):
    qx, mx, ex = mxint_hardware(x, 
                                {
                                    "width": q_config.get("data_in_width"),
                                    "exponent_width": q_config.get("data_in_exponent_width"),
                                    "round_bits": 4
                                },
                                q_config.get("data_in_parallelism"))
    qweight, mweight, eweight = mxint_hardware(weight, 
                                {
                                    "width": q_config.get("weight_width"),
                                    "exponent_width": q_config.get("weight_exponent_width"),
                                    "round_bits": 4
                                },
                                q_config.get("weight_parallelism"))
    qbias, mbias, ebias = mxint_hardware(bias, 
                                {
                                    "width": q_config.get("bias_width"),
                                    "exponent_width": q_config.get("bias_exponent_width"),
                                    "round_bits": 4
                                },
                                q_config.get("bias_parallelism"))
    if q_config["enable_internal_width"]:
        qnorm_out = mxint_layer_norm(qx, normalized_shape, qweight, qbias, eps, q_config)
    else:
        qnorm_out = nn.functional.layer_norm(qx, normalized_shape, qweight.reshape(normalized_shape), qbias.reshape(normalized_shape), eps)
    
    qout, mout, eout = mxint_hardware(qnorm_out, 
                                {
                                    "width": q_config.get("data_out_width"),
                                    "exponent_width": q_config.get("data_out_exponent_width"),
                                    "round_bits": 4
                                },
                                q_config.get("data_out_parallelism"))
    return qout

def int_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, s=1, z=0, r=0, bit_width=8):
    """
    Integer-only LayerNorm implementation as described in the paper.
    
    Args:
        x: Input tensor (X_Q in the paper, already quantized)
        normalized_shape: Shape to normalize over
        weight: gamma parameter (γ in the paper)
        bias: beta parameter (β in the paper)
        eps: small constant for numerical stability
        s: scale factor
        z: zero point
        r: PTR (power-of-two representation)
        bit_width: bit width for quantization (b in the paper)
    
    Returns:
        Quantized output tensor (O_Q in the paper)
    """
    # Handle normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    
    # Get dimensions to normalize over
    dim = tuple(range(-len(normalized_shape), 0))
    
    # Phase 1: Shift the quantized activation X_Q with PTR r (Equation 8)
    # X̂_Q = (X_Q - z) << r
    x_hat_q = (x - z) * 2**r
    
    # Calculate C (batch size over normalization dimensions)
    C = 1
    for i in dim:
        C *= x.shape[i]
    
    # Phase 2: Calculate mean and mean of squares (Equation 9)
    # μx ≈ (1/C)∑(X̂_Q·s) → (s/C)T1
    T1 = torch.sum(x_hat_q, dim=dim, keepdim=True)
    mu_x = (s / C) * T1
    
    # μx2 ≈ (1/C)∑(X̂_Q·s)² → (s²/C)T2
    T2 = torch.sum(x_hat_q**2, dim=dim, keepdim=True)
    mu_x2 = (s**2 / C) * T2
    
    # Calculate variance (Equation 10)
    # σx² = μx2 - μx² ≈ (s²/C²)(CT2 - T1²)
    sigma_x_squared = (s**2 / C**2) * (C * T2 - T1**2)
    
    # Calculate sqrt(σx² + ε) ≈ (s/C)√(CT2 - T1²)
    sqrt_var_eps = (s / C) * torch.sqrt(C * T2 - T1**2 + C**2 * eps / s**2)
    
    # Integer-only inference (Equation 11)
    # Define A and B terms
    if weight is not None:
        s_gamma = s * weight
    else:
        s_gamma = s
        
    A = s_gamma / torch.sqrt(sigma_x_squared + eps)
    
    if bias is not None:
        B = bias * torch.sqrt(sigma_x_squared + eps) - weight * mu_x
    else:
        B = -mu_x
    
    # Calculate M values for bit-width adjustments (Equation 12)
    M1 = bit_width - 1 - torch.floor(torch.log2(torch.abs(A)))
    M2 = torch.floor(torch.abs(A) * 2**M1)
    
    # Apply sign and scaling to A
    A_scaled = torch.sign(A) * (M2 / 2**M1)
    
    # Final quantized output (Equation 12)
    O_Q = torch.floor(A_scaled * x_hat_q + B + z)
    
    return O_Q

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
from .int_quant.quant_modules import IntLayerNormImpl, QuantAct

class IntLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 0.00001, elementwise_affine: bool = False, bias: bool = False, q_config=None,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias)
        self.q_config = q_config
        self.quant_act = QuantAct(
            activation_bit=8, 
            act_range_momentum=0.995
            )
        # self.impl = IntLayerNormImpl(normalized_shape, eps, elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        # self.impl.weight=self.weight
        # self.impl.bias=self.bias
        qx, scaling_factor = self.quant_act(x)
        # breakpoint()
        qout = nn.functional.layer_norm(qx, self.normalized_shape, self.weight, self.bias, self.eps)
        # qout, out_scaling_factor = self.impl(qx, scaling_factor)
        return qout


class QuantLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 0.00001, elementwise_affine: bool = False, bias: bool = False, q_config=None,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias)
        self.q_config = q_config
        if q_config.get('quant_type') == 'mxint':
            logger.debug("Using MXIntLayerNorm")
            self.module = MXIntLayerNorm(
                normalized_shape,
                eps,
                elementwise_affine,
                bias,
                q_config=q_config
            )
        elif q_config.get('quant_type') == 'int':
            logger.debug("Using IntLayerNorm")
            self.module = IntLayerNorm(
                normalized_shape,
                eps,
                elementwise_affine,
                bias,
                q_config=q_config)
        else:
            self.module = nn.LayerNorm(normalized_shape, eps, elementwise_affine, bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

