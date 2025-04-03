from functools import partial
from torch import Tensor
import torch
from torch.nn import functional as F

from chop.nn.quantizers import (
    residual_sign_quantizer,
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    integer_floor_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
    ternary_quantizer,
    mxint_quantizer,
)


class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(x: Tensor, weight: Tensor, bias: Tensor = None):
        return F.linear(x, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inputs, weight, bias = inputs
        ctx.save_for_backward(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(inputs)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# Linearly Quantized recevied output gradient applied to the quantized input, quantized weight
def linearGradInteger(ctx, grad_output, config: dict = None):
    inputs, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    output_grad_width, output_grad_frac_width = (
        config["output_grad_width"],
        config["output_grad_frac_width"],
    )
    input_width, input_frac_width = (
        config["data_in_width"],
        config["data_in_frac_width"],
    )
    weight_width, weight_frac_width = (
        config["weight_width"],
        config["weight_frac_width"],
    )

    floor = config.get("floor", False)
    base_quantizer = integer_floor_quantizer if floor else integer_quantizer

    w_quantizer = partial(
        base_quantizer, width=weight_width, frac_width=weight_frac_width
    )
    x_quantizer = partial(
        base_quantizer, width=input_width, frac_width=input_frac_width
    )
    out_grad_quantizer = partial(
        base_quantizer, width=output_grad_width, frac_width=output_grad_frac_width
    )

    grad_output = out_grad_quantizer(grad_output)
    inputs = x_quantizer(inputs)
    weight = w_quantizer(weight)

    if ctx.needs_input_grad[0]:
        grad_input = grad_output.mm(weight)
    if ctx.needs_input_grad[1]:
        grad_weight = grad_output.t().mm(inputs)
    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_output.sum(0)

    return grad_input, grad_weight, grad_bias
