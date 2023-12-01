#!/usr/bin/env python3
# NOTE: This is not really a test, but a script to just informally validate
# functionality via trial and error. Feel free to modify this file as needed.

import logging
import os
import sys
from pathlib import Path

import toml

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[4].as_posix())

import chop.models as models
from chop.passes import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    conv_bn_fusion_transform_pass,
    add_software_metadata_analysis_pass,
)
from chop.passes.graph.mase_graph import MaseGraph
from chop.tools.get_input import get_dummy_input
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")
logger = logging.getLogger("chop.test")


def main():
    MODEL = "resnet18"
    DATASET = "cifar10"

    # NOTE: We're only concerned with pre-trained vision models
    dataset_info = get_dataset_info(DATASET)
    data_module = MaseDataModule(
        model_name=MODEL,
        name=DATASET,
        batch_size=32,
        num_workers=os.cpu_count(),
        tokenizer=None,
        max_token_len=None,
    )
    data_module.prepare_data()
    data_module.setup()
    # NOTE: We only support vision classification models for now.
    model_info = models.get_model_info(MODEL)
    dummy_input = get_dummy_input(model_info, data_module, "cls")

    model = models.get_model(
        MODEL, task="cls", dataset_info=dataset_info, pretrained=True
    )
    model.eval()  # NOTE: This is a requirement for fusion to work

    graph = MaseGraph(model=model)
    # NOTE: Both functions have pass arguments that are not used in this example
    graph = init_metadata_analysis_pass(graph, None)
    graph = add_common_metadata_analysis_pass(graph, dummy_input)
    graph = add_software_metadata_analysis_pass(graph, None)
    logger.debug(graph.fx_graph)

    # The default save directory is specified as the current working directory
    graph = conv_bn_fusion_transform_pass(graph)


if __name__ == "__main__":
    main()
