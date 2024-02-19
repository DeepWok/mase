import torch
from torch import Tensor
from torch.nn import functional as F


# class BatchNorm1dFixed(torch.nn.BatchNorm1d):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bias: bool = False,
#         device=None,
#         dtype=None,
#     ) -> None:
#         super().__init__(in_features, out_features, bias, device, dtype)
#         self.bypass = False
#         self.x_quantizer = None
#         self.w_quantizer = None
#         self.b_quantizer = None
#         self.pruning_masks = None

#     def forward(self, x: Tensor) -> Tensor:
#         if self.bypass:
#             # if bypss, there is no quantization
#             return F.batch_norm()
#             return F.linear(x, self.weight, self.bias)
#         else:
#             x = self.x_quantizer(x)
#             w = self.w_quantizer(self.weight)
#             bias = self.b_quantizer(self.bias) if self.bias is not None else None
#             return F.linear(x, w, bias)
