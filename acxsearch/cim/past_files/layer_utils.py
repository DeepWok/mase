import torch
import torch.nn as nn
import torch.nn.functional as F
from .int_quant import scale_integer_quantizer

class ReLUNoise(nn.ReLU):
    def __init__(self, q_config={}):
        super(ReLUNoise, self).__init__()
        self.num_bits = q_config.get("num_bits")
        if q_config.get("quantile") is not None:
            self.quantile = q_config.get("quantile")
        else:
            self.quantile = 1.0

    def forward(self, x):
        act = F.relu(x)
        act = scale_integer_quantizer(act, self.num_bits, False, self.quantile)
        return act

class Conv2dNoise(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 q_config={}):
        super(Conv2dNoise, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.q_config = q_config
        self.quantile = q_config.get("quantile")
        self.num_bits = q_config.get("num_bits")
            
    def forward(self, x):
        weights = self.weight
        bias = self.bias

        if self.num_bits is not None:
            weights = scale_integer_quantizer(weights, self.num_bits, False, self.quantile)
            if bias is not None:
                bias = scale_integer_quantizer(bias, self.num_bits, False, self.quantile)

         
        noise_magnitude = self.q_config.get("noise_magnitude")

        if noise_magnitude is not None and noise_magnitude > 0:
            w_max = torch.max(torch.abs(weights))
            weights = weights + torch.randn_like(weights) * w_max * noise_magnitude
            if bias is not None:
                bias = bias + torch.randn_like(bias) * w_max * noise_magnitude
            
        return F.conv2d(x, weights, bias, stride=self.stride, padding=self.padding)

class LinearNoise(nn.Linear):
    def __init__(self, in_features, out_features, q_config={}):
        super(LinearNoise, self).__init__(in_features, out_features)
        self.q_config = q_config
        self.num_bits = q_config.get("num_bits")
        self.noise_magnitude = q_config.get("noise_magnitude")
        self.quantile = q_config.get("quantile")
            
    def forward(self, x):
        weights = self.weight
        if self.bias is not None:
            bias = self.bias
        else:
            bias = torch.tensor(None)
        
        if self.num_bits is not None:
            # Clamp weights and bias
            weights = scale_integer_quantizer(weights, self.num_bits, False, self.quantile)
            if bias is not None:
                bias = scale_integer_quantizer(bias, self.num_bits, False, self.quantile)
            
        noise_magnitude = self.noise_magnitude
        if noise_magnitude is not None and noise_magnitude > 0:
            w_max = torch.max(torch.abs(weights))
            weights = weights + torch.randn_like(weights) * w_max * noise_magnitude
            bias = bias + torch.randn_like(bias) * w_max * noise_magnitude
        return F.linear(x, weights, bias)

class NoiseInjection(nn.Module):
    def __init__(self, noise_std):
        super(NoiseInjection, self).__init__()
        self.noise_std = noise_std
        
    def forward(self, x, training=None):
        if training:
            x = x + torch.randn_like(x) * self.noise_std
        return x 