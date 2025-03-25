"""
Quantization implementation for Wav2Vec2 optimization.
"""

import torch.nn as nn
import logging
from copy import deepcopy
from chop.tools.utils import deepsetattr

# Set up logging
logger = logging.getLogger(__name__)

def apply_quantization(model, quant_method_name, quant_class, bit_config=None):
    """Apply quantization to the model based on specified parameters"""
    logger.info(f"Applying {quant_method_name} quantization")
    
    if quant_method_name == "full_precision":
        return model
    
    # Make a copy of the model
    quant_model = deepcopy(model)
    
    # Apply quantization to each linear layer
    for name, layer in quant_model.named_modules():
        if isinstance(layer, nn.Linear):
            kwargs = {"in_features": layer.in_features, "out_features": layer.out_features}
            
            # Use the provided bit config or default settings
            if bit_config is None:
                if quant_method_name == "integer":
                    config = {
                        "weight_width": 8,
                        "weight_frac_width": 8,
                        "data_in_width": 8,
                        "data_in_frac_width": 8,
                        "bias_width": 16,
                        "bias_frac_width": 8,
                    }
                elif quant_method_name in ["minifloat_denorm", "minifloat_ieee"]:
                    config = {
                        "weight_width": 8,
                        "weight_exponent_width": 5,
                        "weight_exponent_bias": 15,
                        "data_in_width": 8,
                        "data_in_exponent_width": 5,
                        "data_in_exponent_bias": 15,
                        "bias_width": 16,
                        "bias_exponent_width": 5,
                        "bias_exponent_bias": 15,
                    }
                else:
                    # Default config for other quantization methods
                    config = {
                        "weight_width": 8,
                        "data_in_width": 8,
                    }
            else:
                config = bit_config
            
            # Create new quantized layer
            new_layer = quant_class(**kwargs, config=config)
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            
            # Replace the original layer with the quantized one
            deepsetattr(quant_model, name, new_layer)
    
    return quant_model
