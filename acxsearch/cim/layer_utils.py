import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationQuant(nn.Module):
    def __init__(self, num_bits, max_value, decay=0):
        super(ActivationQuant, self).__init__()
        self.num_bits = num_bits
        self.max_value = max_value
        self.decay = decay
        
        if self.num_bits is not None:
            self.relux = nn.Parameter(torch.tensor(max_value))
            
    def forward(self, x):
        if self.num_bits is not None:
            act = torch.clamp(x, min=0, max=int(self.relux))
            # Quantize to num_bits
            scale = (2 ** self.num_bits - 1) / int(self.relux)
            act = torch.round(act * scale) / scale
        else:
            act = F.relu(x)
        return act

class QuantBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, q_config={}):
        super(QuantBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.q_config = q_config
        self.activation_quant = ActivationQuant(num_bits=q_config.get("num_bits"), max_value=q_config.get("max_value"))

    def forward(self, x):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.activation_quant(x)
        return x
        

class Conv2dNoise(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 q_config={}):
        super(Conv2dNoise, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.q_config = q_config
        self.num_bits = q_config.get("num_bits")
        self.weight_range = q_config.get("weight_range")
        self.bias_range = q_config.get("bias_range")
        self.range_decay = q_config.get("range_decay")
        self.noise_magnitude = q_config.get("noise_magnitude")

        if self.num_bits is not None:
            self.weight_range = nn.Parameter(torch.tensor(self.weight_range))
            
    def forward(self, x):
        weights = self.weight
        bias = self.bias
        
        if self.num_bits is not None:
            # Clamp weights and bias
            weights = torch.clamp(weights, -self.weight_range, self.weight_range)
            if bias is not None:
                bias = torch.clamp(bias, -self.bias_range, self.bias_range)
            
            # Quantize weights
            scale = (2 ** self.num_bits - 1) / (2 * self.weight_range)
            weights = torch.round(weights * scale) / scale
            
        noise_magnitude = self.noise_magnitude
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
        self.weight_range = q_config.get("weight_range")
        self.bias_range = q_config.get("bias_range")
        self.range_decay = q_config.get("range_decay")
        self.noise_magnitude = q_config.get("noise_magnitude")
        
        if self.num_bits is not None:
            self.weight_range = nn.Parameter(torch.tensor(self.weight_range))
            
    def forward(self, x):
        weights = self.weight
        if self.bias is not None:
            bias = self.bias
        else:
            bias = torch.tensor(None)
        
        if self.num_bits is not None:
            # Clamp weights and bias
            weights = torch.clamp(weights, -self.weight_range, self.weight_range)
            bias = torch.clamp(bias, -self.bias_range, self.bias_range)
            
            # Quantize weights
            scale = (2 ** self.num_bits - 1) / (2 * self.weight_range)
            weights = torch.round(weights * scale) / scale
            
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