import torch

class HWPQ_PruningOnly:
    def __init__(self, structured_sparsity=False):
        """
        Initialize pruning
        
        Args:
            structured_sparsity: Whether to use 2:4 structured sparsity
        """
        self.structured_sparsity = structured_sparsity
        
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
    
    def prune_weights(self, weights, sparsity_level=0.5):
        """
        Apply pruning to weights without quantization
        
        Args:
            weights: The weight tensor to be pruned
            sparsity_level: Target sparsity level (0.0 to 1.0)
            
        Returns:
            Pruned tensor and mask
        """
        # Make a copy of the weights to modify
        pruned_weights = weights.clone()
        mask = torch.ones_like(weights, dtype=torch.bool)
        
        # Track statistics
        total_weights = weights.numel()
        total_kept = 0
        
        
        # Process each row independently
        for i in range(weights.shape[0]):
            row_weights = weights[i].flatten()
            
            # Get contribution metrics
            contributions, S = self.compute_contribution(row_weights)
            
            # Create a mask for this row
            row_mask = torch.ones_like(row_weights, dtype=torch.bool)
            
            # Count for achieving target sparsity
            n_weights = row_weights.numel()
            target_prune = int(n_weights * sparsity_level)
            
            if target_prune >= n_weights:
                # Avoid pruning all weights
                target_prune = max(0, n_weights - 1)
            
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
            
            # Apply pruning (just set to zero)
            result = torch.zeros_like(row_weights)
            result[row_mask] = row_weights[row_mask]
            
            # Count non-zeros after pruning
            kept_weights = row_mask.sum().item()
            
            
            # Update total statistics
            total_kept += kept_weights
            
            # Save the result and reshape the mask
            pruned_weights[i] = result.reshape(weights[i].shape)
            mask[i] = row_mask.reshape(weights[i].shape)
        

        
        return pruned_weights, mask

def hwpq_pruning_only(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """
    Pruning-only ranking function.
    
    Args:
        tensor: Weight tensor to be pruned
        info: Dictionary with metadata for the tensor
        sparsity: Target sparsity level
        
    Returns:
        Boolean mask indicating which weights to keep (True) or prune (False)
    """
    pruner = HWPQ_PruningOnly()
    
    # Apply pruning to get pruned weights and mask
    _, mask = pruner.prune_weights(tensor, sparsity)
    
    return mask

# For use with pruned modules
class HWPQParameterization(torch.nn.Module):
    """
    Parametrization for pruning only.
    """
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        
    def forward(self, x):
        assert self.mask.shape == x.shape
        # Simply apply the mask for pruning
        pruned = self.mask * x
        return pruned
        
    def state_dict(self, *args, **kwargs):
        # Avoid double saving masks
        return {}