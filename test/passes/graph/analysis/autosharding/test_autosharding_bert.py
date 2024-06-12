import sys, pdb, traceback, os

import torch
import torch.nn as nn

from chop.ir import MaseGraph
from chop.distributed import MaseLauncher
import chop.passes as passes
from chop.tools import get_logger

from chop.models.patched.bert import BertConfig, BertModel
from chop.models.patched.bert.modeling_bert import BertSelfAttention

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

# * Define custom ops (leaf submodules during tracing)
BERT_CUSTOM_OPS = {
    "modules": {
        BertSelfAttention: {},
    },
    "functions": {},
}

def test_autosharding():
    
    # Define config
    config = BertConfig()
    config.num_hidden_layers = 3
    config.hidden_size = 96
    config.intermediate_size = 384
    config_sequence_length = 4

    # Initialize model and MaseGraph
    model = BertModel(config, custom_ops=BERT_CUSTOM_OPS)
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
