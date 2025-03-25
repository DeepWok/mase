"""
Pruning implementation for Wav2Vec2 optimization.
"""

import torch
import logging
from copy import deepcopy
from chop import MaseGraph
import chop.passes as passes

# Set up logging
logger = logging.getLogger(__name__)

def apply_pruning(model, pruning_method, sparsity, structured_sparsity=False):
    """Apply pruning to the model based on specified parameters"""
    logger.info(f"Applying {pruning_method} pruning with sparsity {sparsity}")
    
    # Make a copy of the model
    pruned_model = deepcopy(model)
    
    # Create pruning config
    pruning_config = {
        "weight": {
            "sparsity": sparsity,
            "method": pruning_method,
            "scope": "local",
            "structured_sparsity": structured_sparsity
        },
        "activation": {
            "sparsity": 0.0,
            "method": "random",
            "scope": "local",
        },
    }
    
    # Create temporary MaseGraph for this model instance
    temp_mg = MaseGraph(
        pruned_model,
        hf_input_names=["input_values", "attention_mask"],
    )
    
    # Initialize metadata
    temp_mg, _ = passes.init_metadata_analysis_pass(temp_mg)
    
    # Create dummy input
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    
    # Add common metadata
    temp_mg, _ = passes.add_common_metadata_analysis_pass(
        temp_mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": False,
        }
    )
    
    # Apply pruning transform pass
    temp_mg, _ = passes.prune_transform_pass(temp_mg, pass_args=pruning_config)
    
    return temp_mg.model

def calculate_pruning_metrics(model):
    """Calculate pruning metrics (sparsity, parameter counts)"""
    total_params = 0
    nonzero_params = 0
    pruned_params = 0
    
    # Count parameters using parametrizations (for masked models)
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            for p in module.parametrizations.weight:
                if hasattr(p, 'mask'):
                    total_in_layer = module.weight.numel()
                    nonzero_in_layer = p.mask.sum().item()
                    pruned_in_layer = total_in_layer - nonzero_in_layer
                    
                    pruned_params += pruned_in_layer
                    total_params += total_in_layer
    
    # If no pruning parametrizations found, count zeros in weights
    if total_params == 0:
        for name, param in model.named_parameters():
            if 'weight' in name and 'parametrizations' not in name:
                total_params += param.numel()
                nonzero_params += (param != 0).sum().item()
        
        pruned_params = total_params - nonzero_params
    else:
        nonzero_params = total_params - pruned_params
    
    # Calculate overall sparsity
    overall_sparsity = pruned_params / total_params if total_params > 0 else 0
    
    return {
        "total_weight_params": total_params,
        "nonzero_weight_params": nonzero_params,
        "pruned_weight_params": pruned_params,
        "overall_sparsity": overall_sparsity
    }
