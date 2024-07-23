import sys, pdb, traceback
import pytest

import torch
import torch.nn as nn

from chop.ir import MaseGraph
from chop.distributed import MaseLauncher
import chop.passes as passes
from chop.tools import get_logger

from transformers.models.bert import BertConfig, BertModel

logger = get_logger(__name__)
logger.setLevel("DEBUG")

WORLD_SIZE = 8
DEVICE_MESH = [[0, 1, 2, 3], [4, 5, 6, 7]]


@pytest.mark.skip(reason="Fixing needed")
def test_autosharding():

    # Define config
    config = BertConfig()
    config.num_hidden_layers = 3
    config.hidden_size = 96
    config.intermediate_size = 384
    config._attn_implementation = "eager"
    config_sequence_length = 4

    # Initialize model and MaseGraph
    model = BertModel(config)
    mg = MaseGraph(model)
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.report_graph_analysis_pass(mg, pass_args={"file_name": "bert.txt"})
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": {
                "input_ids": torch.randint(0, 10, (1, config_sequence_length)),
            },
            "add_value": False,
        },
    )

    # Run autosharding pass to decide sharding configuration
    mg, module_map = passes.autosharding_analysis_pass(
        mg,
        pass_args={
            "mesh_shape": (2, 4),
            "inter_node_bandwidth": 10e9,
            "intra_node_bandwidth": 100e9,
        },
    )

    # Insert resharding wrappers around each module to handle inter-operator communication
    mg, _ = passes.resharding_transform_pass(
        mg, pass_args={"module_map": module_map, "device_mesh": DEVICE_MESH}
    )

    # dump print model to a file
    with open("model.txt", "w") as f:
        print(mg.model, file=f)

    # Launch model in distributed cluster
    launcher = MaseLauncher(mg, world_size=WORLD_SIZE, device_mesh=DEVICE_MESH)
    inputs = [torch.randint(0, 10, (1, config_sequence_length))]
    launcher.run(module_map, inputs)


if __name__ == "__main__":
    test_autosharding()
