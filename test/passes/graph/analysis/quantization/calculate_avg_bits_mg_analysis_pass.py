import logging
import os
import sys

import torch

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "..",
        "machop",
    )
)

print(sys.path)

from chop.tools.logger import set_logging_verbosity
from chop.ir.graph import MaseGraph
from chop.models.toys.toy import ToyNet
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    calculate_avg_bits_mg_analysis_pass,
)

logger = logging.getLogger("chop.test")
set_logging_verbosity("debug")


def test_calculate_avg_bits():
    mlp = ToyNet(image_size=(1, 28, 28), num_classes=10)
    mg = MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 8
    x = torch.randn((batch_size, 28 * 28))
    logger.debug(mg.fx_graph)

    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, {})
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"add_value": False, "dummy_in": dummy_in}
    )
    mg, info = calculate_avg_bits_mg_analysis_pass(mg, {})
    print(info)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
test_calculate_avg_bits()
