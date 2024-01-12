import torch
import torch.nn as nn

import logging

import chop
import chop.passes as passes
from chop.tools.logger import set_logging_verbosity

logger = logging.getLogger("chop")
set_logging_verbosity("info")


class MLP(torch.nn.Module):
    """
    Toy model of n linear layers
    """

    def __init__(self, num_layers) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(28 * 28, 28 * 28))

        self.layers.append(nn.Linear(28 * 28, 10))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        x = self.layers[-1](x)

        return x


def test_apply_random_partition():
    num_layers = 10
    num_devices = 5

    mlp = MLP(num_layers=num_layers)
    mg = chop.MaseGraph(model=mlp)

    mg, _ = passes.init_metadata_analysis_pass(mg, None)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 28, 28))
    pass_args = {"dummy_in": {"x": x}}

    mg, _ = passes.add_common_metadata_analysis_pass(mg, pass_args)

    # add metadata for hardware in each mase node of graph
    mg, _ = passes.add_hardware_metadata_analysis_pass(mg)

    mg, _ = passes.partition_to_multi_device_transform_pass(mg)

    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print


if __name__ == "__main__":
    test_apply_random_partition()
