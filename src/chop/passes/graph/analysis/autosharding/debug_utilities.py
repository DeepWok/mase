

import torch.nn as nn

from chop.tools import get_logger

from chop import MaseGraph
import chop.passes as passes
import torch

logger = get_logger(__name__)
logger.setLevel("DEBUG")

def are_layers_equal(layer1, layer2):
    # Check if both layers are instances of the same class
    if type(layer1) != type(layer2):
        return False
    
    # Compare their attributes
    for attr in dir(layer1):
        # Skip methods and special attributes
        l1_attr = getattr(layer1, attr)
        if callable(getattr(layer1, attr)) or attr.startswith("_") or isinstance(l1_attr, torch.Tensor):
            continue
        # Check if both layers have the same attribute and their values are equal
        if hasattr(layer2, attr):
            if getattr(layer1, attr) != getattr(layer2, attr):
                return False
        else:
            return False

    return True

def debug_shardings(layer, input_shardings, world_size, device_mesh):

    from chop.distributed import MaseLauncher

    class WrapperModule(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return self.layer(x)

    logger.info(f"Generating subgraph for layer: {layer}")
    mg = MaseGraph(WrapperModule(layer))
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": {
                "x": torch.randn((1, layer.in_features)),
            },
            "add_value": False,
        },
    )
    
    for idx, sharding in enumerate(input_shardings):
        module_map = {
            "node": "---",
            "sharding": {
                layer: {
                    key: sharding[key] for key in sharding.keys()
                }
            }
        }
        logger.info(f"[{idx}/{len(input_shardings)}] Testing shading: {sharding}")
        launcher = MaseLauncher(mg, world_size=world_size, device_mesh=device_mesh)
        # inputs = [torch.randint(0, 10, (1, config_sequence_length))]
        inputs = [torch.randn((1, layer.in_features))]
        launcher.run(module_map, inputs)