#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report,
    verify_common_metadata_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.logger import set_logging_verbosity

from chop.models.yolo.yolov8 import get_yolo_detection_model


set_logging_verbosity("debug")


def test_yolov8():
    model_name = "yolov8n-seg.pt"
    yolo = get_yolo_detection_model(model_name)
    mg = MaseGraph(model=yolo)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 3, 640, 640))
    dummy_in = {"x": x}

    mg, _ = init_metadata_analysis_pass(mg, {})
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )
    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    quan_args = {
        "by": "type",
        "default": {
            "config": {
                "name": "integer",
                # data
                "data_in_width": 8,
                "data_in_frac_width": 4,
                # weight
                "weight_width": 8,
                "weight_frac_width": 4,
                # bias
                "bias_width": 8,
                "bias_frac_width": 4,
            },
        },
    }

    # deep copy is only possible if we put "add_value" to False
    ori_mg = deepcopy_mase_graph(mg)
    mg, _ = quantize_transform_pass(mg, quan_args)

    pass_args = {
        "original_graph": ori_mg,
        "save_dir": "quantize_summary",
    }
    summarize_quantization_analysis_pass(mg, pass_args)


test_yolov8()
