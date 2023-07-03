#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join("..", "..", "..", "machop"))

from chop.dataset import get_dataset_info
from chop.models import model_map
from chop.passes.analysis.report import report_graph_analysis_pass
from chop.passes.analysis.verify import verify
from chop.passes.graph.mase_graph import MaseGraph


# --------------------------------------------------
#   Emit Verilog using Mase
# --------------------------------------------------
def main():
    load_pretrained = True

    cifar10_info = get_dataset_info("cifar10")

    # MobileNetV2
    mobilenetv2 = model_map["mobilenetv2"](
        info=cifar10_info, pretrained=load_pretrained
    )

    mg = MaseGraph(model=mobilenetv2)
    print(mg.fx_graph)

    # You can compute the mase graph like nn.module:
    # batch_size = 1
    # x = torch.randn((batch_size, 28, 28))
    # print(mg.model(x))

    # Sanity check and report
    # mg = quantize(mg)
    # mg = verify(mg)
    # mg = report(mg)
    # mg = emit_verilog(mg)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
