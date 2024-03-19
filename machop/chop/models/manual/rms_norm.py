import torch
from torch import nn, Tensor


def _rms_norm(x: Tensor, eps, bias, scale, offset):
    mean_squares = x.square().mean(dim=(1, 2, 3), keepdim=True)
    rms_x = mean_squares.sqrt()
    x_normed = x / (rms_x + eps)
    if bias:
        return scale * x_normed + offset
    return scale * x_normed


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-8,
        elementwise_affine: bool = False,
        bias: bool = False,
        device = None,
        dtype = None,
    ):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.bias = bias

        factory_kwargs = {'device': device, 'dtype': dtype}
        if self.elementwise_affine:
            self.scale = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
            if bias:
                self.offset = nn.Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('scale', None)
        else:
            self.register_parameter('scale', None)
            self.register_parameter('offset', None)

    def forward(self, x: Tensor):
        if self.elementwise_affine:
            offset = 0.0 if self.bias else self.offset
            return _rms_norm(x, self.eps, self.bias, self.scale, offset)
        return _rms_norm(x, self.eps, False, 1.0, 0.0)
