import torch
import torch.nn as nn

def test_linear_out_shape(hidden_size=768, out_size=1024):
    """
    Passes a [1, 7, hidden_size] tensor through nn.Linear 
    and prints input/output shapes.
    """
    # Sample input tensor
    x = torch.randn(1, 7, hidden_size)
    
    # Linear layer: change dims if needed
    linear_layer = nn.Linear(hidden_size, out_size)
    
    # Forward pass
    y = linear_layer(x)
    
    # Print shapes for quick verification
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

test_linear_out_shape()