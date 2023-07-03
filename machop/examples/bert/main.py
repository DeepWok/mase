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

    # BERT-small
    # !: not traceable
    sst2_info = get_dataset_info("sst2")
    bert = model_map["bert-base-uncased"](
        name="bert-base-uncased", task="lm", info=sst2_info, pretrained=load_pretrained
    )
    mg = MaseGraph(model=bert)
    print(mg.fx_graph)

    # Sanity check and report
    # mg = quantize(mg)
    # mg = verify(mg)
    # mg = report(mg)
    # mg = emit_verilog(mg)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
