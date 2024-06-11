import torch
import torch.nn as nn

from chop.ir import MaseGraph
import chop.passes as passes

import sys, pdb, traceback


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# Set the custom exception hook
sys.excepthook = excepthook


class MLP(nn.Module):
    def __init__(self, in_features=64, hidden_dimension=128, out_features=64):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_dimension)
        self.l2 = nn.Linear(hidden_dimension, out_features)

    def forward(self, x):
        out = self.l1(x)
        return self.l2(out)


def test_autosharding():
    model = MLP()
    mg = MaseGraph(model)
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg, pass_args={"dummy_in": {"x": torch.randn((16, 64))}, "add_value": False}
    )
    mg, _ = passes.autosharding_analysis_pass(
        mg, 
        pass_args = {
            "mesh_shape": (2, 4),
            "inter_node_bandwidth": 10e9,
            "intra_node_bandwidth": 100e9
        })


if __name__ == "__main__":
    test_autosharding()
