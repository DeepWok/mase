import torch
import torch.nn as nn
import torch.nn.functional as F
from .core.matmul import cim_core

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
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class CIMConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, q_config=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.q_config = {} if q_config is None else q_config

    def forward(self, x):
        B, C, H, W = x.shape
        # 统一 kernel/stride/padding 为 tuple
        kH, kW = (self.kernel_size,)*2 if isinstance(self.kernel_size, int) else self.kernel_size
        sH, sW = (self.stride,)*2     if isinstance(self.stride, int)     else self.stride
        pH, pW = (self.padding,)*2    if isinstance(self.padding, int)    else self.padding

        # 1) unfold -> [B, C*kH*kW, L]
        x_unfold = F.unfold(x, kernel_size=(kH, kW), padding=(pH, pW), stride=(sH, sW))

        weight_flat = self.weight.view(self.out_channels, -1)

        patches = x_unfold.transpose(1, 2).contiguous()        # [B, L, C*kH*kW]
        patches = patches.view(-1, weight_flat.size(1))        # [B*L, C*kH*kW]

        out_flat = cim_core(patches, weight_flat.t(), self.q_config)  
        # -> [B*L, out_channels]

        # reshape back to [B, out_channels, L]
        out = out_flat.view(B, -1, self.out_channels).permute(0, 2, 1)

        H_out = (H + 2*pH - kH) // sH + 1
        W_out = (W + 2*pW - kW) // sW + 1
        output = out.view(B, self.out_channels, H_out, W_out)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output