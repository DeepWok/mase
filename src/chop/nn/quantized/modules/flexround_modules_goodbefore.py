# import torch
# import torch.nn as nn
# from chop.passes.graph.transforms.quantize.flexround import FlexRoundQuantizer

# class LinearFlexRound(nn.Linear):
#     """
#     A linear layer wrapped with FlexRound quantization.
#     """
#     def __init__(self, in_features, out_features, bias=True, config=None):
#         super().__init__(in_features, out_features, bias=bias)
#         # Use the config to extract the quantization parameters.
#         # Here we assume the config uses keys "weight_width" and "weight_frac_width", etc.
#         weight_width = config.get("weight_width", 8)
#         weight_frac = config.get("weight_frac_width", 4)
#         data_in_width = config.get("data_in_width", 8)
#         data_in_frac = config.get("data_in_frac_width", 4)
#         bias_width = config.get("bias_width", 8)
#         bias_frac = config.get("bias_frac_width", 4)
#         weight_only = config.get("weight_only", False)
        
#         self.weight_quant = FlexRoundQuantizer(bit_width=weight_width, frac_width=weight_frac)
#         # Optionally quantize activations if not weight-only.
#         if not weight_only:
#             self.act_quant = FlexRoundQuantizer(bit_width=data_in_width, frac_width=data_in_frac)
#         else:
#             self.act_quant = None

#     def forward(self, input):
#         # Quantize weight (note: in practice you might want to wrap weight or use a hook)
#         quant_w = self.weight_quant(self.weight)
#         out = nn.functional.linear(input, quant_w, self.bias)
#         # Optionally quantize activations
#         if self.act_quant is not None:
#             out = self.act_quant(out)
#         return out

# # Similarly for conv2d:
# class Conv2dFlexRound(nn.Conv2d):
#     """
#     A Conv2d layer wrapped with FlexRound quantization.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', config=None):
#         super().__init__(in_channels, out_channels, kernel_size, stride,
#                          padding, dilation, groups, bias, padding_mode)
#         weight_width = config.get("weight_width", 8)
#         weight_frac = config.get("weight_frac_width", 4)
#         data_in_width = config.get("data_in_width", 8)
#         data_in_frac = config.get("data_in_frac_width", 4)
#         bias_width = config.get("bias_width", 8)
#         bias_frac = config.get("bias_frac_width", 4)
#         weight_only = config.get("weight_only", False)
        
#         self.weight_quant = FlexRoundQuantizer(bit_width=weight_width, frac_width=weight_frac)
#         if not weight_only:
#             self.act_quant = FlexRoundQuantizer(bit_width=data_in_width, frac_width=data_in_frac)
#         else:
#             self.act_quant = None

#     def forward(self, x):
#         quant_w = self.weight_quant(self.weight)
#         out = self._conv_forward(x, quant_w, self.bias)
#         if self.act_quant is not None:
#             out = self.act_quant(out)
#         return out


# class Conv1dFlexRound(nn.Conv1d):
#     """
#     A Conv1d layer wrapped with FlexRound quantization.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, config=None):
#         super().__init__(in_channels, out_channels, kernel_size, stride,
#                          padding, dilation, groups, bias)
#         # Use the config to set quantization parameters.
#         weight_width = config.get("weight_width", 8)
#         weight_frac = config.get("weight_frac_width", 4)
#         data_in_width = config.get("data_in_width", 8)
#         data_in_frac = config.get("data_in_frac_width", 4)
#         bias_width = config.get("bias_width", 8)
#         bias_frac = config.get("bias_frac_width", 4)
#         weight_only = config.get("weight_only", False)
        
#         self.weight_quant = FlexRoundQuantizer(bit_width=weight_width, frac_width=weight_frac)
#         if not weight_only:
#             self.act_quant = FlexRoundQuantizer(bit_width=data_in_width, frac_width=data_in_frac)
#         else:
#             self.act_quant = None

#     def forward(self, x):
#         quant_w = self.weight_quant(self.weight)
#         out = nn.functional.conv1d(x, quant_w, self.bias, self.stride,
#                                      self.padding, self.dilation, self.groups)
#         if self.act_quant is not None:
#             out = self.act_quant(out)
#         return out



"""
File: src/chop/nn/quantized/modules/flexround_modules.py

This file defines operator‐specific FlexRound wrappers for quantization.
Each class subclasses the corresponding PyTorch module (e.g. nn.Linear, nn.Conv2d, nn.Conv1d)
and applies FlexRound quantization to the weights (and optionally activations) during forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chop.passes.graph.transforms.quantize.flexround import FlexRoundQuantizer

class LinearFlexRound(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, config=None):
        """
        A linear layer wrapped with FlexRound quantization.

        The config should contain:
          - "weight_width", "weight_frac_width"
          - "data_in_width", "data_in_frac_width"
          - "bias_width", "bias_frac_width"
          - "weight_only" (bool)
        """
        super().__init__(in_features, out_features, bias=bias)
        config = config or {}
        # Use the standard key names expected by the parser.
        weight_width = config.get("weight_width", 8)
        weight_frac  = config.get("weight_frac_width", 4)
        data_in_width = config.get("data_in_width", 8)
        data_in_frac  = config.get("data_in_frac_width", 4)
        bias_width = config.get("bias_width", 8)
        bias_frac  = config.get("bias_frac_width", 4)
        weight_only = config.get("weight_only", False)
        self.weight_quantizer = FlexRoundQuantizer(bit_width=weight_width, frac_width=weight_frac)
        # If not weight-only, quantize activations as well.
        if not weight_only:
            self.act_quantizer = FlexRoundQuantizer(bit_width=data_in_width, frac_width=data_in_frac)
        else:
            self.act_quantizer = None

    def forward(self, input):
        # Quantize the weight
        quant_w = self.weight_quantizer(self.weight)
        out = F.linear(input, quant_w, self.bias)
        # Optionally quantize the activation
        if self.act_quantizer is not None:
            out = self.act_quantizer(out)
        return out

class Conv2dFlexRound(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros",
                 config=None):
        """
        A Conv2d layer wrapped with FlexRound quantization.

        Expects config with keys:
          - "weight_width", "weight_frac_width"
          - "data_in_width", "data_in_frac_width"
          - "bias_width", "bias_frac_width"
          - "weight_only" (bool)
        """
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
        config = config or {}
        weight_width = config.get("weight_width", 8)
        weight_frac  = config.get("weight_frac_width", 4)
        data_in_width = config.get("data_in_width", 8)
        data_in_frac  = config.get("data_in_frac_width", 4)
        bias_width = config.get("bias_width", 8)
        bias_frac  = config.get("bias_frac_width", 4)
        weight_only = config.get("weight_only", False)
        self.weight_quantizer = FlexRoundQuantizer(bit_width=weight_width, frac_width=weight_frac)
        if not weight_only:
            self.act_quantizer = FlexRoundQuantizer(bit_width=data_in_width, frac_width=data_in_frac)
        else:
            self.act_quantizer = None

    def forward(self, x):
        quant_w = self.weight_quantizer(self.weight)
        out = self._conv_forward(x, quant_w, self.bias)
        if self.act_quantizer is not None:
            out = self.act_quantizer(out)
        return out

class Conv1dFlexRound(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, config=None):
        """
        A Conv1d layer wrapped with FlexRound quantization.

        Expects config with keys:
          - "weight_width", "weight_frac_width"
          - "data_in_width", "data_in_frac_width"
          - "bias_width", "bias_frac_width"
          - "weight_only" (bool)
        """
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        config = config or {}
        weight_width = config.get("weight_width", 8)
        weight_frac  = config.get("weight_frac_width", 4)
        data_in_width = config.get("data_in_width", 8)
        data_in_frac  = config.get("data_in_frac_width", 4)
        bias_width = config.get("bias_width", 8)
        bias_frac  = config.get("bias_frac_width", 4)
        weight_only = config.get("weight_only", False)
        self.weight_quantizer = FlexRoundQuantizer(bit_width=weight_width, frac_width=weight_frac)
        if not weight_only:
            self.act_quantizer = FlexRoundQuantizer(bit_width=data_in_width, frac_width=data_in_frac)
        else:
            self.act_quantizer = None

    def forward(self, x):
        quant_w = self.weight_quantizer(self.weight)
        out = F.conv1d(x, quant_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.act_quantizer is not None:
            out = self.act_quantizer(out)
        return out
