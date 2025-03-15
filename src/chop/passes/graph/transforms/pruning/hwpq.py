import torch
import math

class FP8Format:
    """
    Represents FP8 (E5M2) format - 5 bits exponent, 2 bits mantissa, 1 bit sign
    """
    def __init__(self):
        self.exponent_bits = 5
        self.mantissa_bits = 2
        self.sign_bit = 1
        
    def quantize(self, tensor):
        """
        Quantize a tensor to FP8 (E5M2) format with safeguards
        """
        # Create a copy of the tensor to avoid modifying the original
        fp8_tensor = tensor.clone()
        
        # Skip quantization for small tensors
        if tensor.numel() < 10:
            return tensor
            
        # Calculate the mean and std of the tensor for scale-aware quantization
        tensor_mean = torch.mean(torch.abs(tensor))
        if tensor_mean < 1e-5:
            # If values are too small, preserve them without quantization
            return tensor
            
        # Apply the quantization constraints of FP8
        # Limit the range based on exponent bits
        max_exp = 2**(2**(self.exponent_bits-1)-1)
        min_exp = -max_exp
        
        # Clamp values to the representable range
        fp8_tensor = torch.clamp(fp8_tensor, min_exp, max_exp)
        
        # Quantize the mantissa to 2 bits of precision
        # This is a simplified implementation - real hardware would do this differently
        scale = 2**(math.floor(math.log2(max_exp)) - self.mantissa_bits)
        
        # Only quantize values above a threshold to prevent small values from going to zero
        threshold = scale * 0.5
        mask = torch.abs(fp8_tensor) >= threshold
        fp8_tensor[mask] = torch.round(fp8_tensor[mask] / scale) * scale
        
        # Verify we haven't zeroed out the entire tensor
        if (fp8_tensor != 0).sum().item() == 0 and (tensor != 0).sum().item() > 0:
            print("WARNING: FP8 quantization zeroed out all values! Falling back to original values.")
            return tensor
            
        return fp8_tensor

class HWPQ:
    def __init__(self, alpha=0.125, beta=0.125, la=4.0, structured_sparsity=False):
        """
        Initialize HWPQ with parameters for EWMA
        
        Args:
            alpha: EWMA parameter for estimating mean
            beta: EWMA parameter for estimating deviation
            la: Threshold multiplier for pruning decision
            structured_sparsity: Whether to use 2:4 structured sparsity
        """
        self.alpha = alpha
        self.beta = beta
        self.la = la
        self.structured_sparsity = structured_sparsity
        self.fp8 = FP8Format()
        
    def compute_contribution(self, weights):
        """
        Compute contribution metric L = w_i^2 / (1 - x_i^2/S) for each weight
        
        Args:
            weights: Tensor of weights in a single layer
            
        Returns:
            Tensor of contribution metrics for each weight
        """
        # Compute S (sum of squared weights)
        S = torch.sum(weights**2)
        
        # Compute the contribution metric for each weight
        x_squared = weights**2
        denominator = 1 - x_squared / (S + 1e-10)  # Add epsilon to avoid division by zero
        # Avoid division by zero or negative values
        denominator = torch.clamp(denominator, min=1e-10)
        
        contributions = x_squared / denominator
        return contributions, S
    
    def prune_and_quantize_weights(self, weights, sparsity_level=0.5):
        """
        Apply HWPQ to prune and quantize weights
        
        Args:
            weights: The weight tensor to be pruned and quantized
            sparsity_level: Target sparsity level (0.0 to 1.0)
            
        Returns:
            Pruned and quantized tensor and mask
        """
        # Make a copy of the weights to modify
        pruned_weights = weights.clone()
        mask = torch.ones_like(weights, dtype=torch.bool)
        
        # Process each row independently
        for i in range(weights.shape[0]):
            row_weights = weights[i].flatten()
            
            # Get contribution metrics
            contributions, _ = self.compute_contribution(row_weights)
            
            # Create a mask for this row
            row_mask = torch.ones_like(row_weights, dtype=torch.bool)
            
            # Count for achieving target sparsity
            n_weights = row_weights.numel()
            target_prune = int(n_weights * sparsity_level)
            
            if target_prune >= n_weights:
                # Avoid pruning all weights
                target_prune = max(0, n_weights - 1)
            
            if self.structured_sparsity and n_weights >= 4:  # Only apply if we have enough weights
                # For 2:4 structured sparsity
                pruned_count = 0
                
                for start_idx in range(0, n_weights, 4):
                    end_idx = min(start_idx + 4, n_weights)
                    chunk_size = end_idx - start_idx
                    
                    if chunk_size < 4:  # Handle incomplete chunks differently
                        if chunk_size > 1:  # If at least 2 weights, prune proportionally
                            prune_in_chunk = max(1, int(chunk_size * 0.5))  # Prune ~50%
                            group_contrib = contributions[start_idx:end_idx]
                            _, indices = torch.topk(group_contrib, prune_in_chunk, largest=False)
                            indices = indices + start_idx
                            row_mask[indices] = 0
                    else:
                        # Full chunk of 4 - prune 2
                        group_contrib = contributions[start_idx:end_idx]
                        _, indices = torch.topk(group_contrib, 2, largest=False)
                        indices = indices + start_idx
                        row_mask[indices] = 0
            else:
                # For unstructured pruning
                # Sort contributions to ensure we prune exactly the target number
                sorted_indices = torch.argsort(contributions)
                
                # Prune the weights with lowest contributions up to target sparsity
                prune_indices = sorted_indices[:target_prune]
                row_mask[prune_indices] = 0
            
            # Ensure we're not pruning everything
            if (row_mask == 0).all():
                # Keep at least one weight (the one with highest contribution)
                max_idx = torch.argmax(contributions)
                row_mask[max_idx] = 1
            
            # Quantize the remaining weights to FP8
            result = torch.zeros_like(row_weights)
            result[row_mask] = self.fp8.quantize(row_weights[row_mask])
            
            # Save the result and reshape the mask
            pruned_weights[i] = result.reshape(weights[i].shape)
            mask[i] = row_mask.reshape(weights[i].shape)
        
        return pruned_weights, mask

def hwpq_pruning(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """
    HWPQ pruning ranking function for MASE pruning framework.
    
    Args:
        tensor: Weight tensor to be pruned
        info: Dictionary with metadata for the tensor
        sparsity: Target sparsity level
        
    Returns:
        Boolean mask indicating which weights to keep (True) or prune (False)
    """
    structured_sparsity = info.get("structured_sparsity", True)
    hwpq = HWPQ(structured_sparsity=structured_sparsity)
    
    # Apply HWPQ to get pruned weights and mask
    _, mask = hwpq.prune_and_quantize_weights(tensor, sparsity)
    
    return mask

# For use with quantized modules
class HWPQParameterization(torch.nn.Module):
    """
    Parametrization for HWPQ. This applies both pruning and FP8 quantization.
    """
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.fp8 = FP8Format()
        
    def forward(self, x):
        assert self.mask.shape == x.shape
        pruned = self.mask * x
        quantized = self.fp8.quantize(pruned)
        return quantized
        
    def state_dict(self, *args, **kwargs):
        # Avoid double saving masks
        return {}