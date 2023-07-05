#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join("..", "..", "..", "..", "machop"))

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
    # OPT
    # !: not traceable
    wikitext_info = get_dataset_info("wikitext2")
    opt_dict = model_map["facebook/opt-125m"](
        name="facebook/opt-125m",
        task="lm",
        info=wikitext_info,
        pretrained=load_pretrained,
    )
    opt = opt_dict["model"]
    opt_tokenizer = opt_dict["tokenizer"]
    mg = MaseGraph(model=opt)
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
