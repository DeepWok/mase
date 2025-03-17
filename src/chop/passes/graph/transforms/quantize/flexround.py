import torch
import math
import torch.nn as nn

class FlexRoundQuantizer(nn.Module):
    """
    FlexRound quantization module for MASE
    
    This module implements the FlexRound quantization technique, which uses
    a fixed-point representation with configurable bit-width and fraction width.
    """
    def __init__(self, bit_width=8, frac_width=4, symmetric=True):
        """
        Initialize FlexRound quantizer with configurable parameters.
        
        Args:
            bit_width (int): Total number of bits for quantization
            frac_width (int): Number of fractional bits for fixed-point representation
            symmetric (bool): Whether to use symmetric quantization
        """
        super().__init__()
        self.bit_width = bit_width
        self.frac_width = frac_width
        self.symmetric = symmetric
        self.scale = 2.0 ** frac_width
        
        # Calculate quantization bounds
        if symmetric:
            self.min_val = -2**(bit_width-1) / self.scale
            self.max_val = (2**(bit_width-1) - 1) / self.scale
        else:
            self.min_val = 0
            self.max_val = (2**bit_width - 1) / self.scale
            
    def forward(self, x):
        """
        Apply FlexRound quantization to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor to quantize
            
        Returns:
            torch.Tensor: Quantized tensor
        """
        # Skip quantization for very small tensors
        if x.numel() < 10:
            return x
            
        # Check if tensor has enough significant values to quantize
        if x.abs().max() < 1e-5:
            return x
            
        # Clamp values to representable range
        x_clamped = torch.clamp(x, self.min_val, self.max_val)
        
        # Apply fixed-point quantization
        scaled = x_clamped * self.scale
        rounded = torch.round(scaled)
        quantized = rounded / self.scale
        
        return quantized
        
    def get_config(self):
        """Return the configuration parameters of the quantizer"""
        return {
            "bit_width": self.bit_width,
            "frac_width": self.frac_width,
            "symmetric": self.symmetric,
            "scale": self.scale,
            "min_val": self.min_val,
            "max_val": self.max_val
        }

class FlexRoundQuantizedLayer(nn.Module):
    """
    A module wrapper that applies FlexRound quantization to a layer's weights and/or activations
    """
    def __init__(self, 
                 module, 
                 weight_width=8, 
                 weight_frac_width=4, 
                 data_in_width=8, 
                 data_in_frac_width=4,
                 weight_only=False):
        """
        Initialize a FlexRound quantized layer.
        
        Args:
            module (nn.Module): The module to quantize
            weight_bit_width (int): Bit width for weight quantization
            weight_frac_width (int): Fraction width for weight quantization
            act_bit_width (int): Bit width for activation quantization
            act_frac_width (int): Fraction width for activation quantization
            weight_only (bool): If True, only quantize weights (not activations)
        """
        super().__init__()
        self.module = module
        self.weight_quantizer = FlexRoundQuantizer(weight_width, weight_frac_width)
        
        if not weight_only:
            self.act_quantizer = FlexRoundQuantizer(data_in_width, data_in_frac_width)
        else:
            self.act_quantizer = None
            
        # Initialize to apply quantization
        self.apply_weight_quant = True
        self.apply_act_quant = not weight_only
        
    def forward(self, x):
        """Apply quantization to weights and forward pass"""
        # Backup original weights
        if hasattr(self.module, 'weight'):
            orig_weight = self.module.weight.data.clone()
            
            # Apply weight quantization
            if self.apply_weight_quant:
                self.module.weight.data = self.weight_quantizer(orig_weight)
                
            # Run forward pass
            out = self.module(x)
            
            # Restore original weights
            self.module.weight.data = orig_weight
        else:
            out = self.module(x)
        
        # Apply activation quantization if enabled
        if self.apply_act_quant and self.act_quantizer is not None:
            out = self.act_quantizer(out)
            
        return out
        
    def set_quant_state(self, weight_quant=None, act_quant=None):
        """Enable/disable quantization for weights and/or activations"""
        if weight_quant is not None:
            self.apply_weight_quant = weight_quant
        if act_quant is not None:
            self.apply_act_quant = act_quant

def quantize_model_with_flexround(model, config):
    """
    Apply FlexRound quantization to a model based on the provided configuration.
    
    Args:
        model (nn.Module): The model to quantize
        config (dict): Configuration specifying how to quantize the model
        
    Returns:
        nn.Module: The quantized model
    """
    # Create a copy of the model to avoid modifying the original
    quantized_model = type(model)(model.config) if hasattr(model, 'config') else model
    
    # Default configuration
    default_config = config.get('default', {})
    default_weight_bit_width = default_config.get('weight_width', 8)
    default_weight_frac_width = default_config.get('weight_frac_width', 4)
    default_act_bit_width = default_config.get('data_in_width', 8)
    default_act_frac_width = default_config.get('data_in_frac_width', 4)
    default_weight_only = default_config.get('weight_only', False)
    
    # Track quantized modules
    quantized_modules = {}
    
    # Replace modules with quantized versions according to configuration
    for name, module in model.named_modules():
        # Skip the top level module
        if name == '':
            continue
            
        # Check if this module has a specific configuration
        if name in config:
            module_config = config[name]
            weight_bit_width = module_config.get('weight_width', default_weight_bit_width)
            weight_frac_width = module_config.get('weight_frac_width', default_weight_frac_width)
            act_bit_width = module_config.get('data_in_width', default_act_bit_width)
            act_frac_width = module_config.get('data_in_frac_width', default_act_frac_width)
            weight_only = module_config.get('weight_only', default_weight_only)
            
            # Replace with quantized version if it has weights
            if hasattr(module, 'weight'):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                
                parent = model if parent_name == '' else getattr(model, parent_name)
                
                quantized_module = FlexRoundQuantizedLayer(
                    module,
                    weight_frac_width,
                    act_bit_width,
                    act_frac_width,
                    weight_only
                )
                
                # Replace the module
                setattr(parent, child_name, quantized_module)
                quantized_modules[name] = quantized_module
    
    print(f"Applied FlexRound quantization to {len(quantized_modules)} modules")
    return quantized_model


# For use with the MASE framework
def apply_flexround_transform(graph, config):
    """
    Apply FlexRound quantization to a MASE graph.
    
    Args:
        graph: The MASE graph to apply quantization to
        config: Configuration dict with quantization parameters
        
    Returns:
        The quantized graph
    """
    # Default config
    default_config = config.get('default', {})
    weight_bit_width = default_config.get('weight_bit_width', 8)
    weight_frac_width = default_config.get('weight_frac_width', 4)
    
    # Apply quantization to each module
    for node in graph.fx_graph.nodes:
        if node.op == 'call_module':
            module = graph.modules[node.target]
            module_name = node.target
            
            # Get module-specific config if available
            if module_name in config:
                module_config = config[module_name]
                this_weight_bit_width = module_config.get('weight_bit_width', weight_bit_width)
                this_weight_frac_width = module_config.get('weight_frac_width', weight_frac_width)
            else:
                this_weight_bit_width = weight_bit_width
                this_weight_frac_width = weight_frac_width
            
            # Check if the module has weights to quantize
            if hasattr(module, 'weight'):
                # Apply quantization by wrapping the module weight parameter
                quantizer = FlexRoundQuantizer(
                    bit_width=this_weight_bit_width,
                    frac_width=this_weight_frac_width
                )
                
                # Create a forward hook to quantize weights during inference
                def make_quant_hook(mod, quantizer):
                    def hook(module, inputs):
                        module.weight.data = quantizer(module.weight.data)
                        return None
                    return hook
                
                # Register the pre-forward hook to quantize weights
                module.register_forward_pre_hook(make_quant_hook(module, quantizer))
                
                print(f"Applied FlexRound quantization to {module_name} with {this_weight_bit_width} bits")
    
    return graph, {}