import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_logger


# Set up logger
logger = get_logger(__name__)
logger.setLevel("INFO")

def count_nonzero_parameters(model):
    """Count the actual non-zero parameters in the model"""
    total_params = 0
    nonzero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and 'parametrizations' not in name:
            # Count total parameters
            total_params += param.numel()
            
            # Count non-zero parameters
            nonzero_params += (param != 0).sum().item()
                
    return total_params, nonzero_params

def print_parameter_count(model, description):
    """Helper function to count and print parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Also count non-zero parameters
    total, nonzero = count_nonzero_parameters(model)
    sparsity = 1.0 - (nonzero / total) if total > 0 else 0
    
    print(f"\n===== {description} =====")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total weight parameters: {total:,}")
    print(f"Non-zero weight parameters: {nonzero:,}")
    print(f"Sparsity: {sparsity:.2%}")
    
    return total_params, nonzero, sparsity

def main():
    print("\n===== HWPQ Pruning Example =====")
    
    # Load a pretrained model
    checkpoint = "facebook/wav2vec2-base-960h"
    model = AutoModelForCTC.from_pretrained(checkpoint)
    encoder = model.wav2vec2
    
    # Create a MASE graph
    mg = MaseGraph(encoder, hf_input_names=["input_values", "attention_mask"])
    mg, _ = passes.init_metadata_analysis_pass(mg)
    
    # Define dummy input for analysis pass
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    mg, _ = passes.add_common_metadata_analysis_pass(mg,
                                                   pass_args={
                                                       "dummy_in": dummy_in,
                                                       "add_value": True,
                                                       "force_device_meta": False,
                                                   })
    
    # Print initial parameter count
    before_params, before_nonzero, _ = print_parameter_count(mg.model, "BEFORE PRUNING")
    
    # Configure HWPQ pruning
    hwpq_config = {
        "weight": {
            "sparsity": 0.1,           # 50% sparsity
            "method": "hwpq",          # Use our HWPQ method
            "scope": "local",          # Apply locally per layer
            "structured_sparsity": False # Use 2:4 structured sparsity
        },
        "activation": {
            "sparsity": 0.0,           # No activation pruning for HWPQ
            "method": "random",
            "scope": "local",
        },
    }
    
    print("\n===== APPLYING HWPQ PRUNING =====")
    mg, _ = passes.prune_transform_pass(mg, pass_args=hwpq_config)
    
    # Count pruned parameters by examining the parametrizations
    pruned_params = 0
    total_weight_params = 0
    
    for name, module in mg.model.named_modules():
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            for p in module.parametrizations.weight:
                if hasattr(p, 'mask'):
                    # This is our pruning parametrization
                    weight_shape = module.weight.shape
                    total_in_layer = module.weight.numel()
                    nonzero_in_layer = p.mask.sum().item()
                    pruned_in_layer = total_in_layer - nonzero_in_layer
                    
                    print(f"Layer {name}: pruned {pruned_in_layer}/{total_in_layer} params ({pruned_in_layer/total_in_layer:.2%})")
                    
                    pruned_params += pruned_in_layer
                    total_weight_params += total_in_layer
    
    if total_weight_params > 0:
        overall_sparsity = pruned_params / total_weight_params
        print(f"\nOverall from HWPQ masks: {pruned_params}/{total_weight_params} params pruned ({overall_sparsity:.2%} sparsity)")
    
    # Print parameter stats after pruning
    after_params, after_nonzero, after_sparsity = print_parameter_count(mg.model, "AFTER PRUNING")
    
    # Calculate and print change in parameters
    print(f"\n===== PRUNING SUMMARY =====")
    print(f"Parameters before pruning:   {before_params:,}")
    print(f"Non-zero params before:      {before_nonzero:,}")
    print(f"Non-zero params after:       {after_nonzero:,}")
    print(f"Reduction in parameters:     {before_nonzero - after_nonzero:,}")
    print(f"Overall sparsity achieved:   {after_sparsity:.2%}")
    
    return mg.model

if __name__ == "__main__":
    main()