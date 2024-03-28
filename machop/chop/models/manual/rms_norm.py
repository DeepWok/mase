import torch
from torch import nn, Tensor


def _rms_norm(x: Tensor, eps, scale: Tensor | None):
    mean_squares = x.square().mean(dim=(1, 2, 3), keepdim=True)
    rms_x = mean_squares.sqrt()
    x_normed = x / (rms_x + eps)
    if scale != None:
        return scale * x_normed
    else:
        return x_normed


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-8,
        elementwise_affine: bool = False,
        device = None,
        dtype = None,
    ):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine

        factory_kwargs = {'device': device, 'dtype': dtype}
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: Tensor):
        return _rms_norm(x, self.eps, self.weight)
