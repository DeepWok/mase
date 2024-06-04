
import torch
import torch.nn as nn

from chop.ir import MaseGraph
import chop.passes as passes

class MLP(nn.Module):
    def __init__(self, in_features = 64, hidden_dimension=512, out_features=64):
        super().__init__()
        self.l1 = nn.Linear(in_features = in_features, out_features=hidden_dimension)
        self.l2 = nn.Linear(in_features = hidden_dimension, out_features=hidden_dimension)
        self.l3 = nn.Linear(in_features = hidden_dimension, out_features=out_features)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        return out
    
def test_autosharding():
    model = MLP()
    model, _ = passes.autosharding_analysis_pass(model)

if __name__ == "__main__":
    test_autosharding()