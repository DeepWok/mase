#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import toml
import torch
import torch.nn as nn


from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())
from chop.models import get_model_info
from chop.models.toys.toy_custom_fn import ToyCustomFnNet
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)

from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.get_input import InputGenerator
from chop.dataset import MaseDataModule
from chop.tools.logger import set_logging_verbosity

from chop.tools.utils import device, to_numpy_if_tensor

set_logging_verbosity("debug")

logger = logging.getLogger("chop.test")


def test_metadata_value_type_cast():
    batch_size = 8
    mlp = ToyCustomFnNet(image_size=(3, 32, 32), num_classes=10, batch_size=batch_size)

    # Provide a dummy input for the graph so it can use for tracing
    x = torch.randn((batch_size, 3 * 32 * 32))
    mlp(x)

    dummy_in = {"x": x}

    data_module = MaseDataModule(
        model_name="toy-fn",
        name="cifar10",
        batch_size=8,
        num_workers=0,
        tokenizer=None,
        max_token_len=128,
    )
    data_module.prepare_data()
    data_module.setup()

    model_info = get_model_info("toy_custom_fn")
    input_generator = InputGenerator(
        model_info=model_info,
        data_module=data_module,
        task="cls",
        which_dataloader="train",
    )

    mg = MaseGraph(model=mlp)
    logger.debug(mg.fx_graph)
    mg, _ = init_metadata_analysis_pass(mg, None)
    # mg = add_mase_ops_analysis_pass(mg, dummy_in)
    mg, _ = add_common_metadata_analysis_pass(
        mg, pass_args={"dummy_in": dummy_in, "add_value": False}
    )
    mg, _ = add_software_metadata_analysis_pass(mg, pass_args=None)

    mg, _ = metadata_value_type_cast_transform_pass(
        mg, pass_args={"fn": to_numpy_if_tensor}
    )


test_metadata_value_type_cast()
