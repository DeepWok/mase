import torch
import torch.nn as nn
import torch.nn.functional as F
from .core.matmul import cim_core
import math


class LoraCIMLinear(nn.Linear):
    """
    Linear layer with cim simulation and LoRA adapter.
    lora_config should be a dict with the following keys:
    - r: int
    - lora_alpha: float
    - lora_dropout: float
    - adapter_name: str
    - disable_adapter: bool

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        q_config=None,
        lora_config=None,
    ):
        self.q_config = {} if q_config is None else q_config
        super().__init__(in_features, out_features, bias)
        lora_r = lora_config["r"]
        self.lora_A = nn.Parameter(torch.zeros(in_features, lora_r))
        self.lora_B = nn.Parameter(torch.zeros(lora_r, out_features))
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
        # self.weight.requires_grad = False
        self.scaling = lora_config["lora_alpha"] / lora_r

    def _linear(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output = cim_core(input, weight.t(), self.q_config)

        # Add bias if provided
        if self.bias is not None:
            output = output + self.bias
        return output

    def forward(self, x: torch.Tensor):
        delta_weight = (self.lora_A @ self.lora_B * self.scaling).transpose(0, 1)
        new_weight = self.weight + delta_weight
        result = self._linear(x, new_weight)

        return result


class CIMLinear(nn.Linear):
    """
    Linear layer with PCM noise simulation.
    Similar to nn.Linear but with noise modeling for PCM-based computation.

    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True
        config (dict, optional): Configuration for noise parameters
    """

    def __init__(self, in_features, out_features, bias=True, q_config=None):
        super(CIMLinear, self).__init__(in_features, out_features, bias)
        self.q_config = {} if q_config is None else q_config

    def forward(self, input):
        # Apply noisy matrix multiplication using the custom autograd function
        output = cim_core(input, self.weight.t(), self.q_config)

        # Add bias if provided
        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class CIMConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        q_config=None,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.q_config = {} if q_config is None else q_config

    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = (
            (self.kernel_size,) * 2
            if isinstance(self.kernel_size, int)
            else self.kernel_size
        )
        sH, sW = (self.stride,) * 2 if isinstance(self.stride, int) else self.stride
        pH, pW = (self.padding,) * 2 if isinstance(self.padding, int) else self.padding

        # 1) unfold -> [B, C*kH*kW, L]
        x_unfold = F.unfold(x, kernel_size=(kH, kW), padding=(pH, pW), stride=(sH, sW))

        weight_flat = self.weight.view(self.out_channels, -1)

        patches = x_unfold.transpose(1, 2).contiguous()  # [B, L, C*kH*kW]
        patches = patches.view(-1, weight_flat.size(1))  # [B*L, C*kH*kW]

        out_flat = cim_core(patches, weight_flat.t(), self.q_config)
        # -> [B*L, out_channels]

        # reshape back to [B, out_channels, L]
        out = out_flat.view(B, -1, self.out_channels).permute(0, 2, 1)

        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1
        output = out.view(B, self.out_channels, H_out, W_out)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output
