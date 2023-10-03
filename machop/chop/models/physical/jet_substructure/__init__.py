"""
Jet Substructure Models used in the LogicNets paper
"""

import torch.nn as nn


class JSC_S(nn.Module):
    def __init__(self, info):
        super(JSC_S, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
        )

    def forward(self, x):
        return self.seq_blocks(x.view(x.size(0), -1))


class JSC_Full(nn.Module):
    def __init__(self, info):
        super(JSC_Full, self).__init__()
        self.config = info
        self.num_features = self.config["num_features"]
        self.num_classes = self.config["num_classes"]
        hidden_layers = [64, 32, 32, 32]
        self.num_neurons = [self.num_features] + hidden_layers + [self.num_classes]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            layer = []
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                in_act = nn.Hardtanh()
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [bn_in, in_act, fc, bn, out_act]
            elif i == len(self.num_neurons) - 1:
                in_act = nn.Hardtanh()
                fc = nn.Linear(in_features, out_features)
                out_act = nn.Hardtanh()
                layer = [fc, bn, out_act]
            else:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, out_act]
            layer_list = layer_list + layer
        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x


# Getters ------------------------------------------------------------------------------
def get_jsc_s(info):
    # return JSC_S()
    return JSC_Full(info)


def get_jsc_full(info):
    # TODO: Tanh is not supported by mase yet
    return JSC_S(info)
