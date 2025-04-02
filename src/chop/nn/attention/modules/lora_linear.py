from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=None):
        super().__init__()
        if rank is None:
            rank = min(in_features, out_features) // 4
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.B(self.A(x))
