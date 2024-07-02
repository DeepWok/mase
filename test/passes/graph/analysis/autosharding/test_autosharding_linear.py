import sys, pdb, traceback, os

import torch
import torch.nn as nn

from chop.ir import MaseGraph
from chop.distributed import MaseLauncher
import chop.passes as passes
from chop.tools import get_logger

def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# Set the custom exception hook
sys.excepthook = excepthook

logger = get_logger(__name__)
logger.setLevel("DEBUG")

WORLD_SIZE = 8
DEVICE_MESH = [[0, 1, 2, 3], [4, 5, 6, 7]]

class MLP(nn.Module):
    def __init__(self, in_features=64, hidden_dimension=128, out_features=64):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_dimension)
        self.l2 = nn.Linear(hidden_dimension, out_features)

    def forward(self, x):
        out = self.l1(x)
        return self.l2(out)

def test_autosharding():
    
    # Initialize model and MaseGraph
    model = MLP()
    mg = MaseGraph(model)
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg, pass_args={"dummy_in": {"x": torch.randn((16, 64))}, "add_value": False}
    )

    # Run autosharding pass to decide sharding configuration
    mg, module_map = passes.autosharding_analysis_pass(
        mg, 
        pass_args = {
            "mesh_shape": (2, 4),
            "inter_node_bandwidth": 10e9,
            "intra_node_bandwidth": 100e9
        })

    # Insert resharding wrappers around each module to handle inter-operator communication
    mg, _ = passes.resharding_transform_pass(mg, pass_args={"module_map": module_map, "device_mesh": DEVICE_MESH})

    # Launch model in distributed cluster
    launcher = MaseLauncher(mg, world_size=WORLD_SIZE, device_mesh=DEVICE_MESH)
    inputs = [torch.randn((16, 64))]
    launcher.run(module_map, inputs)


if __name__ == "__main__":
    test_autosharding()
