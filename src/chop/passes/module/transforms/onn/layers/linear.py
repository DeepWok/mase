"""
Optical Transformer Linear Layer.

This module provides the optical transformer linear layer implementation
by importing from the mase-triton package.
"""

from mase_triton.optical_compute import layers as OTLayers

#: Optical Transformer Linear layer.
#:
#: This is an alias to ``mase_triton.optical_compute.layers.OpticalTransformerLinear``.
#: It replaces standard ``torch.nn.Linear`` layers with quantized optical transformer
#: equivalents that simulate optical neural network hardware constraints.
#:
#: The layer applies quantization to both the input activations and weights during
#: matrix multiplication, and tracks running min/max statistics for calibration.
#:
#: Use the ``from_linear`` class method to convert an existing ``torch.nn.Linear``:
#:
#: .. code-block:: python
#:
#:     from chop.passes.module.transforms.onn.layers.linear import OtLinear
#:
#:     ot_linear = OtLinear.from_linear(
#:         linear_layer,
#:         q_levels=256,
#:         q_lut_min=0.020040,
#:     )
OtLinear = OTLayers.OpticalTransformerLinear
