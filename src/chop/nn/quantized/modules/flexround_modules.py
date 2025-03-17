# File: src/chop/nn/quantized/modules/flexround_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from chop.passes.graph.transforms.quantize.flexround import FlexRoundQuantizer

#############################################
# Linear FlexRound wrapper
#############################################
class LinearFlexRound(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, config=None):
        """
        A linear layer wrapped with FlexRound quantization.
        Quantized weights are computed as:
            Ŵ = s1 * floor( W / (s1 * S2 * s3) + 0.5 )
        where:
          - s1 is a learnable scalar (common grid size)
          - S2 is a learnable tensor (same shape as weight)
          - s3 is a learnable vector of shape (out_features, 1)
        The config dict should contain keys:
          "weight_width", "weight_frac_width",
          "data_in_width", "data_in_frac_width",
          "bias_width", "bias_frac_width", and optionally "weight_only".
          It may also contain "s1_init" (default=1.0).
        """
        super().__init__(in_features, out_features, bias=bias)
        config = config or {}
        s1_init = config.get("s1_init", 1.0)
        # Initialize learnable parameters for FlexRound:
        self.s1 = nn.Parameter(torch.tensor(s1_init, dtype=self.weight.dtype))
        self.S2 = nn.Parameter(torch.ones_like(self.weight))
        # s3 is a per–output–channel scaling factor (shape: [out_features, 1])
        self.s3 = nn.Parameter(torch.ones(self.out_features, 1, dtype=self.weight.dtype))
        
        # Set up activation quantizer if needed.
        if not config.get("weight_only", False):
            act_width = config.get("data_in_width", 8)
            act_frac  = config.get("data_in_frac_width", 4)
            self.act_quant = FlexRoundQuantizer(bit_width=act_width, frac_width=act_frac)
        else:
            self.act_quant = None

    def forward(self, input):
        # Compute the effective division factor
        div_factor = self.s1 * self.S2 * self.s3
        # Apply FlexRound quantization to weights
        quant_w = self.s1 * torch.floor(self.weight / div_factor + 0.5)
        out = F.linear(input, quant_w, self.bias)
        if self.act_quant is not None:
            out = self.act_quant(out)
        return out

#############################################
# Conv2d FlexRound wrapper
#############################################
class Conv2dFlexRound(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros",
                 config=None):
        """
        A Conv2d layer wrapped with FlexRound quantization.
        Quantized weights are computed as:
            Ŵ = s1 * floor( W / (s1 * S2 * s3 * s4) + 0.5 )
        where:
          - s1 is a learnable scalar,
          - S2 is a learnable tensor (same shape as weight),
          - s3 is a learnable per–output–channel factor (shape: [out_channels, 1, 1, 1]),
          - s4 is a learnable per–input–channel factor (shape: [1, in_channels, 1, 1]).
        """
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
        config = config or {}
        s1_init = config.get("s1_init", 1.0)
        self.s1 = nn.Parameter(torch.tensor(s1_init, dtype=self.weight.dtype))
        self.S2 = nn.Parameter(torch.ones_like(self.weight))
        self.s3 = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1, dtype=self.weight.dtype))
        self.s4 = nn.Parameter(torch.ones(1, self.in_channels, 1, 1, dtype=self.weight.dtype))
        
        if not config.get("weight_only", False):
            act_width = config.get("data_in_width", 8)
            act_frac  = config.get("data_in_frac_width", 4)
            self.act_quant = FlexRoundQuantizer(bit_width=act_width, frac_width=act_frac)
        else:
            self.act_quant = None

    def forward(self, x):
        div_factor = self.s1 * self.S2 * self.s3 * self.s4
        quant_w = self.s1 * torch.floor(self.weight / div_factor + 0.5)
        out = self._conv_forward(x, quant_w, self.bias)
        if self.act_quant is not None:
            out = self.act_quant(out)
        return out

#############################################
# Conv1d FlexRound wrapper
#############################################
class Conv1dFlexRound(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, config=None, **kwargs):
        """
        A Conv1d layer wrapped with FlexRound quantization.
        Quantized weights are computed as:
            Ŵ = s1 * floor( W / (s1 * S2 * s3) + 0.5 )
        where:
          - s1 is a learnable scalar,
          - S2 is a learnable tensor (same shape as weight),
          - s3 is a learnable per–output–channel factor (shape: [out_channels, 1, 1]).
        The **kwargs argument allows ignoring unsupported parameters such as padding_mode.
        """
        # nn.Conv1d does not support padding_mode, so we pop it if present.
        kwargs.pop("padding_mode", None)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, **kwargs)
        config = config or {}
        s1_init = config.get("s1_init", 1.0)
        self.s1 = nn.Parameter(torch.tensor(s1_init, dtype=self.weight.dtype))
        self.S2 = nn.Parameter(torch.ones_like(self.weight))
        self.s3 = nn.Parameter(torch.ones(self.out_channels, 1, 1, dtype=self.weight.dtype))
        
        if not config.get("weight_only", False):
            act_width = config.get("data_in_width", 8)
            act_frac  = config.get("data_in_frac_width", 4)
            self.act_quant = FlexRoundQuantizer(bit_width=act_width, frac_width=act_frac)
        else:
            self.act_quant = None

    def forward(self, x):
        div_factor = self.s1 * self.S2 * self.s3
        quant_w = self.s1 * torch.floor(self.weight / div_factor + 0.5)
        out = F.conv1d(x, quant_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.act_quant is not None:
            out = self.act_quant(out)
        return out

#############################################
# (Optional) Activation FlexRound quantizer
#############################################
# If needed, you can also include a simple FlexRoundQuantizer definition here,
# but it’s already available in the quantize/flexround.py module.
