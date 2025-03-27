"""
Quantization implementation for Wav2Vec2 optimization.
"""

import torch.nn as nn
import logging
from copy import deepcopy
from chop.tools.utils import deepsetattr

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
    
    # For each module (using the full dotted name) in the model
    for name, layer in quant_model.named_modules():
        if isinstance(layer, nn.Linear):
            # For this layer, the available types are all the precision choices
            available_types = [cls.__name__ for cls in precision_choices]
            chosen_type_name = trial.suggest_categorical(f"{name}_type", available_types)
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
                    "weight_width": trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32]),
                    "weight_frac_width": trial.suggest_categorical(f"{name}_weight_frac_width", [4, 8, 16]),
                    "data_in_width": trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32]),
                    "data_in_frac_width": trial.suggest_categorical(f"{name}_data_in_frac_width", [4, 8, 16]),
                    "bias_width": trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32]),
                    "bias_frac_width": trial.suggest_categorical(f"{name}_bias_frac_width", [4, 8, 16]),
                }
            elif chosen_type_name in ["LinearMinifloatDenorm", "LinearMinifloatIEEE"]:
                config = {
                    "weight_width": trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32]),
                    "weight_exponent_width": trial.suggest_categorical(f"{name}_weight_exponent_width", [3, 5, 8]),
                    "weight_exponent_bias": trial.suggest_categorical(f"{name}_weight_exponent_bias", [7, 15, 31]),
                    "data_in_width": trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32]),
                    "data_in_exponent_width": trial.suggest_categorical(f"{name}_data_in_exponent_width", [3, 5, 8]),
                    "data_in_exponent_bias": trial.suggest_categorical(f"{name}_data_in_exponent_bias", [7, 15, 31]),
                    "bias_width": trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32]),
                    "bias_exponent_width": trial.suggest_categorical(f"{name}_bias_exponent_width", [3, 5, 8]),
                    "bias_exponent_bias": trial.suggest_categorical(f"{name}_bias_exponent_bias", [7, 15, 31]),
                }
            elif chosen_type_name == "LinearLog":
                config = {
                    "weight_width": trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32]),
                    "weight_exponent_bias": trial.suggest_categorical(f"{name}_weight_exponent_bias", [0, 7, 15]),
                    "data_in_width": trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32]),
                    "data_in_exponent_bias": trial.suggest_categorical(f"{name}_data_in_exponent_bias", [0, 7, 15]),
                    "bias_width": trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32]),
                    "bias_exponent_bias": trial.suggest_categorical(f"{name}_bias_exponent_bias", [0, 7, 15]),
                }
            elif chosen_type_name == "LinearBlockFP":
                config = {
                    "weight_width": trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32]),
                    "weight_exponent_width": 5,
                    "weight_exponent_bias": 15,
                    "weight_block_size": [16],  # fixed for now
                    "data_in_width": trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32]),
                    "data_in_exponent_width": 5,
                    "data_in_exponent_bias": 15,
                    "data_in_block_size": [16],
                    "data_in_skip_first_dim": True,
                    "bias_width": trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32]),
                    "bias_exponent_width": 5,
                    "bias_exponent_bias": 15,
                    "bias_block_size": [16],
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