import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_logger
from chop.passes.module import report_trainable_parameters_analysis_pass
# Import the FlexRound utilities:
from chop.passes.graph.transforms.quantize.flexround import (
    FlexRoundQuantizer,
    apply_flexround_transform
)
# Import the modular quantization pass – ensure this is the correct path
from chop.passes.graph.transforms.quantize.quantize import quantize_transform_pass


# Ensure that FlexRound keys are in the quantized module map
from chop.nn.quantized.modules import quantized_module_map
from chop.nn.quantized.modules.flexround_modules import LinearFlexRound, Conv2dFlexRound, Conv1dFlexRound

quantized_module_map["linear_flexround"] = LinearFlexRound
quantized_module_map["conv2d_flexround"] = Conv2dFlexRound
quantized_module_map["conv1d_flexround"] = Conv1dFlexRound

from chop.passes.module import report_trainable_parameters_analysis_pass





# Set up logger
logger = get_logger(__name__)
logger.setLevel("INFO")

def count_nonzero_parameters(model):
    """Count the actual non-zero parameters in the model."""
    total_params = 0
    nonzero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name and 'parametrizations' not in name:
            total_params += param.numel()
            nonzero_params += (param != 0).sum().item()
    return total_params, nonzero_params

def print_parameter_count(model, description):
    """Helper function to count and print parameters."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total, nonzero = countnonzero = count_nonzero_parameters(model)
    sparsity = 1.0 - (nonzero / total) if total > 0 else 0
    print(f"\n===== {description} =====")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total weight parameters: {total:,}")
    print(f"Non-zero weight parameters: {nonzero:,}")
    print(f"Sparsity: {sparsity:.2%}")
    return total_params, nonzero, sparsity

def main():
    print("\n===== FlexRound Quantization Example =====")
    
    # Load a pretrained model
    checkpoint = "facebook/wav2vec2-base-960h"
    model = AutoModelForCTC.from_pretrained(checkpoint)
    encoder = model.wav2vec2
    
    # Create a MASE graph from the encoder
    mg = MaseGraph(encoder, hf_input_names=["input_values", "attention_mask"])
    mg, _ = passes.init_metadata_analysis_pass(mg)
    
    # Define dummy input for analysis
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    mg, _ = passes.add_common_metadata_analysis_pass(mg, pass_args={
        "dummy_in": dummy_in,
        "add_value": True,
        "force_device_meta": False,
    })




    # Print initial parameter count
    before_params, before_nonzero, _ = print_parameter_count(mg.model, "BEFORE QUANTIZATION")
    
    # -------------------------
    # Method 1: Direct Module Quantization using FlexRoundQuantizer
    # -------------------------
    print("\n===== METHOD 1: DIRECT MODULE QUANTIZATION =====")
    _, _ = report_trainable_parameters_analysis_pass(mg.model)

    found_module = False
    for name, module in mg.model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            found_module = True
            print(f"Found sample module: {name}")
            
            # Create a quantizer instance
            quantizer = FlexRoundQuantizer(bit_width=8, frac_width=4)
            
            # Get weight statistics before quantization
            weight_mean_before = module.weight.abs().mean().item()
            unique_values_before = torch.unique(module.weight).numel()
            print("Module weight stats before quantization:")
            print(f"  Mean abs value: {weight_mean_before:.6f}")
            print(f"  Number of unique values: {unique_values_before}")
            
            with torch.no_grad():
                # Save original weights
                original_weight = module.weight.clone()
                # Apply quantization directly
                module.weight.data = quantizer(module.weight.data)
                weight_mean_after = module.weight.abs().mean().item()
                unique_values_after = torch.unique(module.weight).numel()
                print("Module weight stats after quantization:")
                print(f"  Mean abs value: {weight_mean_after:.6f}")
                print(f"  Number of unique values: {unique_values_after}")
                print(f"  Reduction in unique values: {unique_values_before/unique_values_after:.2f}x")
                # Restore original weights
                module.weight.data = original_weight
            break
    if not found_module:
        print("Could not find a Linear module with weights for direct quantization")
    
    # -------------------------
    # Method 2: Using the FlexRound API for the entire model
    # -------------------------
    print("\n===== METHOD 2: USING FlexRound QUANTIZATION API =====")
    flexround_config = {
        "default": {
            "weight_bit_width": 8,
            "weight_frac_width": 4
        }
    }
    quantized_graph, _ = apply_flexround_transform(mg, flexround_config)
    after_params, after_nonzero, _ = print_parameter_count(quantized_graph.model, "AFTER QUANTIZATION (Method 2)")
    print("\n===== QUANTIZATION SUMMARY (Method 2) =====")
    print(f"Parameters before quantization:   {before_params:,}")
    print(f"Non-zero params before:           {before_nonzero:,}")
    print(f"Non-zero params after:            {after_nonzero:,}")
    print(f"Change in effective parameters:   {before_nonzero - after_nonzero:,}")
    quantized_modules = sum(1 for _, m in quantized_graph.model.named_modules()
                            if hasattr(m, '_forward_hooks') and len(m._forward_hooks) > 0)
    print(f"Total quantized modules: {quantized_modules}")
    
    # -------------------------
    # Method 3: Using quantize_transform_pass for full-model quantization with FlexRound
    # -------------------------
    print("\n===== METHOD 3: USING quantize_transform_pass =====")
    
    # from chop.passes.graph.transforms.quantize.quant_parsers.update_node_meta import quant_arith_to_list_fn
    # quant_arith_to_list_fn["flexround"] = [
    #     "weight_bit_width", "weight_frac_width",
    #     "act_bit_width", "act_frac_width",
    #     "bias_bit_width", "bias_frac_width",
    # ]
    
    
    quant_config = {
        "by": "type",
        "default": {"config": {"name": None}},  # leave unlisted ops unchanged
        "linear": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
                "weight_only": False,
            }
        },
        "conv2d": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
                "weight_only": False,
            }
        },
        "conv1d": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
                "weight_only": False,
            }
        },
        # Add other op types if needed...
    }


    quantized_graph2, _ = quantize_transform_pass(mg, quant_config)
    final_params, final_nonzero, _ = print_parameter_count(quantized_graph2.model, "AFTER quantize_transform_pass (Method 3)")
    print("\n===== QUANTIZATION SUMMARY (Method 3) =====")
    print(f"Parameters before quantization:   {before_params:,}")
    print(f"Non-zero params before:           {before_nonzero:,}")
    print(f"Non-zero params after:            {final_nonzero:,}")
    print(f"Change in effective parameters:   {before_nonzero - final_nonzero:,}")
    final_quantized_modules = sum(1 for _, m in quantized_graph2.model.named_modules()
                                  if hasattr(m, '_forward_hooks') and len(m._forward_hooks) > 0)
    print(f"Total quantized modules (Method 3): {final_quantized_modules}")
    
    return quantized_graph2.model

if __name__ == "__main__":
    main()
