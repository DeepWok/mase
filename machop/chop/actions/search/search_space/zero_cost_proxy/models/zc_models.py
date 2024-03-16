import torch.nn as nn
import torch.nn.functional as F

# Define the linear model
class ZeroCostLinearModel(nn.Module):
    def __init__(self, input_size):
        super(ZeroCostLinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        # 3 inputs (zc metrics), 1 output (accuracy)

    def forward(self, x):
        return self.linear(x)
    
class ZeroCostNonLinearModel(nn.Module):
    def __init__(self, input_size):
        super(ZeroCostNonLinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 13)  # First linear layer
        self.act1 = nn.ReLU()            # Non-linear activation function
        self.linear2 = nn.Linear(13, 1)  # Second linear layer to produce 1 output

    def forward(self, x):
        x = self.linear1(x)  # Pass input through the first linear layer
        x = self.act1(x)     # Apply non-linear activation
        x = self.linear2(x)  # Pass through the second linear layer
        return x
    