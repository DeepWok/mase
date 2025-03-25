from torch import nn

class ResidualFC(nn.Module):
    def __init__(self, new_fc: nn.Module, original_fc: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.new_fc = new_fc
        self.original_fc = original_fc
        self.alpha = alpha 

    def forward(self, hidden_states, **kwargs):
        new_out = self.new_fc(hidden_states)
        orig_out = self.original_fc(hidden_states)
        output = self.alpha * new_out + (1 - self.alpha) * orig_out
        return output 