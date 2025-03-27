"""
Quantization implementation for Wav2Vec2 optimization.
"""

import torch.nn as nn
import logging
from copy import deepcopy
from chop.tools.utils import deepsetattr

# Import parameter choices from config
from config import (
    WEIGHT_WIDTH_CHOICES, WEIGHT_FRAC_WIDTH_CHOICES,
    DATA_IN_WIDTH_CHOICES, DATA_IN_FRAC_WIDTH_CHOICES,
    BIAS_WIDTH_CHOICES, BIAS_FRAC_WIDTH_CHOICES,
    WEIGHT_EXPONENT_WIDTH_CHOICES, WEIGHT_EXPONENT_BIAS_CHOICES,
    DATA_IN_EXPONENT_WIDTH_CHOICES, DATA_IN_EXPONENT_BIAS_CHOICES,
    BIAS_EXPONENT_WIDTH_CHOICES, BIAS_EXPONENT_BIAS_CHOICES,
    LOG_EXPONENT_BIAS_CHOICES
)

# Set up logging
logger = logging.getLogger(__name__)

def apply_mixed_precision_quantization(model, precision_choices, trial):
    """
    Apply mixed precision quantization to the model, allowing each layer to 
    have its own quantization type and configuration.
    
    Args:
        model: The model to be quantized
        precision_choices: List of available quantization classes
        trial: The current Optuna trial
    
    Returns:
        Quantized model and dictionary of precision decisions per layer
    """
    logger.info("Applying mixed precision quantization")
    
    # Make a copy of the model
    quant_model = deepcopy(model)
    
    # Dictionary to record the decision made per layer
    precision_decisions = {}
    
    # Generate a layer ID map to create unique but consistent parameter names
    layer_id_map = {}
    for idx, (name, _) in enumerate(quant_model.named_modules()):
        if isinstance(_, nn.Linear):
            layer_id_map[name] = f"layer_{idx}"
    
    # For each module (using the full dotted name) in the model
    for name, layer in quant_model.named_modules():
        if isinstance(layer, nn.Linear):
            # Get the short layer id from the map
            layer_id = layer_id_map[name]
            
            # For this layer, the available types are all the precision choices
            available_types = [cls.__name__ for cls in precision_choices]
            chosen_type_name = trial.suggest_categorical(f"{layer_id}_type", available_types)
            precision_decisions[name] = chosen_type_name
            
            logger.info(f"Layer {name} set to type: {chosen_type_name}")
            
            # If full precision is chosen, leave the layer unchanged
            if chosen_type_name == "Linear":
                continue
                
            # Otherwise, pick the corresponding quantized class
            chosen_cls = next(cls for cls in precision_choices if cls.__name__ == chosen_type_name)
            
            # Prepare common kwargs
            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }
            
            # Sample hyperparameters based on the chosen quantized type
            config = {}
            if chosen_type_name == "LinearInteger":
                config = {
                    "weight_width": trial.suggest_categorical(f"integer_weight_width", WEIGHT_WIDTH_CHOICES),
                    "weight_frac_width": trial.suggest_categorical(f"integer_weight_frac_width", WEIGHT_FRAC_WIDTH_CHOICES),
                    "data_in_width": trial.suggest_categorical(f"integer_data_in_width", DATA_IN_WIDTH_CHOICES),
                    "data_in_frac_width": trial.suggest_categorical(f"integer_data_in_frac_width", DATA_IN_FRAC_WIDTH_CHOICES),
                    "bias_width": trial.suggest_categorical(f"integer_bias_width", BIAS_WIDTH_CHOICES),
                    "bias_frac_width": trial.suggest_categorical(f"integer_bias_frac_width", BIAS_FRAC_WIDTH_CHOICES),
                }
            elif chosen_type_name in ["LinearMinifloatDenorm", "LinearMinifloatIEEE"]:
                minifloat_type = "minifloat"  # Common prefix for both types
                config = {
                    "weight_width": trial.suggest_categorical(f"{minifloat_type}_weight_width", WEIGHT_WIDTH_CHOICES),
                    "weight_exponent_width": trial.suggest_categorical(f"{minifloat_type}_weight_exponent_width", WEIGHT_EXPONENT_WIDTH_CHOICES),
                    "weight_exponent_bias": trial.suggest_categorical(f"{minifloat_type}_weight_exponent_bias", WEIGHT_EXPONENT_BIAS_CHOICES),
                    "data_in_width": trial.suggest_categorical(f"{minifloat_type}_data_in_width", DATA_IN_WIDTH_CHOICES),
                    "data_in_exponent_width": trial.suggest_categorical(f"{minifloat_type}_data_in_exponent_width", DATA_IN_EXPONENT_WIDTH_CHOICES),
                    "data_in_exponent_bias": trial.suggest_categorical(f"{minifloat_type}_data_in_exponent_bias", DATA_IN_EXPONENT_BIAS_CHOICES),
                    "bias_width": trial.suggest_categorical(f"{minifloat_type}_bias_width", BIAS_WIDTH_CHOICES),
                    "bias_exponent_width": trial.suggest_categorical(f"{minifloat_type}_bias_exponent_width", BIAS_EXPONENT_WIDTH_CHOICES),
                    "bias_exponent_bias": trial.suggest_categorical(f"{minifloat_type}_bias_exponent_bias", BIAS_EXPONENT_BIAS_CHOICES),
                    "data_in_frac_width": 16,  
                    "bias_frac_width": 16,  
                }
            elif chosen_type_name == "LinearLog":
                config = {
                    "weight_width": trial.suggest_categorical(f"log_weight_width", WEIGHT_WIDTH_CHOICES),
                    "weight_exponent_bias": trial.suggest_categorical(f"log_weight_exponent_bias", LOG_EXPONENT_BIAS_CHOICES),
                    "data_in_width": trial.suggest_categorical(f"log_data_in_width", DATA_IN_WIDTH_CHOICES),
                    "data_in_exponent_bias": trial.suggest_categorical(f"log_data_in_exponent_bias", LOG_EXPONENT_BIAS_CHOICES),
                    "data_in_frac_width": 16,  
                    "bias_width": trial.suggest_categorical(f"log_bias_width", BIAS_WIDTH_CHOICES),
                    "bias_exponent_bias": trial.suggest_categorical(f"log_bias_exponent_bias", LOG_EXPONENT_BIAS_CHOICES),
                    "bias_frac_width": 16,  
                }
            elif chosen_type_name == "LinearBlockFP":
                config = {
                    "weight_width": trial.suggest_categorical(f"blockfp_weight_width", WEIGHT_WIDTH_CHOICES),
                    "weight_exponent_width": 5,
                    "weight_exponent_bias": 15,
                    "weight_block_size": [16],  # fixed for now
                    "data_in_width": trial.suggest_categorical(f"blockfp_data_in_width", DATA_IN_WIDTH_CHOICES),
                    "data_in_exponent_width": 5,
                    "data_in_exponent_bias": 15,
                    "data_in_block_size": [16],
                    "data_in_skip_first_dim": True,
                    "data_in_frac_width": 16,  
                    "bias_width": trial.suggest_categorical(f"blockfp_bias_width", BIAS_WIDTH_CHOICES),
                    "bias_exponent_width": 5,
                    "bias_exponent_bias": 15,
                    "bias_block_size": [16],
                    "bias_frac_width": 16,
                    "weight_frac_width": 16,  
                }
                
            # Create the new (quantized) layer and copy the parameters
            new_layer = chosen_cls(**kwargs, config=config)
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
                
            # Propagate metadata if available
            if hasattr(layer, "meta"):
                new_layer.meta = layer.meta.copy()
            else:
                new_layer.meta = {}
                
            # Replace the layer in the model
            deepsetattr(quant_model, name, new_layer)
    
    # Record the per-layer decisions
    trial.set_user_attr("precision_decisions", precision_decisions)
    
    return quant_model, precision_decisions