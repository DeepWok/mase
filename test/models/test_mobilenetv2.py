#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os
import sys

import torch
import torch.nn as nn

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "machop"
    )
)

from chop.dataset import get_dataset_info
from chop.models import get_model
from chop.passes.graph.analysis.report import report_graph_analysis_pass
from chop.passes.graph.analysis.verify import verify
from chop.ir.graph.mase_graph import MaseGraph


# --------------------------------------------------
#   Emit Verilog using Mase
# --------------------------------------------------
def test_mobilenetv2():
    load_pretrained = True

    cifar10_info = get_dataset_info("cifar10")

    # MobileNetV2
    mobilenetv2 = get_model(
        "mobilenetv2", task="cls", dataset_info=cifar10_info, pretrained=True
    )

    mg = MaseGraph(model=mobilenetv2)
    # print(mg.fx_graph)

    # You can compute the mase graph like nn.module:
    # batch_size = 1
    # x = torch.randn((batch_size, 28, 28))
    # print(mg.model(x))

    # Sanity check and report
    # mg = quantize(mg)
    # mg = verify(mg)
    # mg = report(mg)
    # mg = emit_verilog(mg)
