import torch
from torch import Tensor
import sys
from einops import rearrange

sys.path.append("/workspace/machop/")

import torch
from torch import Tensor
from torch.autograd.function import InplaceFunction
from numpy import ndarray


class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


my_clamp = MyClamp.apply
my_round = MyRound.apply


def _integer_quantize(
    x: Tensor | ndarray, width: int, frac_width: int, is_signed: bool = True
):
    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    # thresh = 2 ** (width - 1)
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_round(x.mul(scale)), int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_round(x * scale), int_min, int_max) / scale


def quantize_to_int(x: Tensor, width: int, frac_width: int):
    x = _integer_quantize(x, width, frac_width) * (2**frac_width)
    x = x.int() & (2**width - 1)
    return x


def tensor_cast(tensor_in, in_width, in_frac_width, out_width, out_frac_width):
    size = torch.tensor(tensor_in.shape)
    tensor_temp = tensor_in.reshape(torch.prod(size))

    for i in range(len(tensor_temp)):
        in_value = int(tensor_temp[i])
        if in_frac_width > out_frac_width:
            in_value = in_value >> (in_frac_width - out_frac_width)
        else:
            in_value = in_value << (out_frac_width - in_frac_width)
        in_int_width = in_width - in_frac_width
        out_int_width = out_width - out_frac_width
        if in_int_width > out_int_width:
            if in_value >> (in_frac_width + out_int_width) > 0:
                in_value = 1 << out_width - 1
            elif in_value >> (in_frac_width + out_int_width) < 0:
                in_value = -(1 << out_width - 1)
            else:
                in_value = int(in_value % (1 << out_width))
        tensor_temp[i] = in_value
    tensor_out = tensor_temp.reshape(list(size))
    return tensor_out


def linear_data_pack(
    samples, in_temp: Tensor, in_y: int, in_x: int, unroll_in_y: int, unroll_in_x: int
):
    np = int(in_y / unroll_in_y)
    d = int(in_x / unroll_in_x)
    p = unroll_in_y
    s = unroll_in_x

    in_temp = in_temp.to(torch.int).reshape(samples, np * p, d * s)
    ref = []
    for i in range(samples):
        re_tensor = rearrange(
            in_temp[i], "(np p) (d s) -> np (p d) s", np=np, d=d, p=p, s=s
        )
        ex_tensor = torch.zeros(np, d * p, s, dtype=int)
        for b in range(np):
            for i in range(d):
                for j in range(p):
                    ex_tensor[b][i * p + j] = re_tensor[b][j * d + i]
        output_tensor = rearrange(
            ex_tensor, "np (d p) s -> (np d) (p s)", np=np, d=d, p=p, s=s
        )
        output = output_tensor.tolist()
        ref = ref + output
    return ref
