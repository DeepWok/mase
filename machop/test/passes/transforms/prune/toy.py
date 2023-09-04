#!/usr/bin/env python3
# A simple example of using the prune transform pass on a toy model for both weight and
# activation pruning. See the accompanying config file for more details.


import logging
import os
import sys
from pathlib import Path

import toml

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[4].as_posix())

from chop.passes import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    prune_transform_pass,
    add_software_metadata_analysis_pass,
)
from chop.models import get_model_info
from chop.models.toys.toy import ToyConvNet
from chop.passes.graph.mase_graph import MaseGraph
from chop.tools.logger import getLogger
from chop.dataset import MaseDataModule
from chop.tools.get_input import InputGenerator, get_dummy_input


logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    BATCH_SIZE = 32

    root = Path(__file__).resolve().parents[5]
    config_path = root / "machop/configs/tests/prune/toy.toml"
    with open(config_path, "r") as f:
        config = toml.load(f)

        data_module = MaseDataModule(
            model_name="toy_conv",  # This doesn't really matter
            name="cifar10",
            batch_size=BATCH_SIZE,
            num_workers=os.cpu_count(),
            tokenizer=None,
            max_token_len=None,
        )
        data_module.prepare_data()
        data_module.setup()
        # NOTE: We only support vision classification models for now.
        model_info = get_model_info("toy_convnet")
        dummy_input = get_dummy_input(model_info, data_module, "cls")

        # We need the input generator to do a sample forward pass to log information on
        # the channel-wise activation sparsity.

        input_generator = InputGenerator(
            model_info=model_info,
            data_module=data_module,
            task="cls",
            which_dataloader="train",
        )

        model = ToyConvNet(num_classes=10)
        graph = MaseGraph(model=model)

        # NOTE: Both functions have pass arguments that are not used in this example
        graph = init_metadata_analysis_pass(graph, None)
        graph = add_common_metadata_analysis_pass(graph, dummy_input)
        graph = add_software_metadata_analysis_pass(graph, None)
        logger.debug(graph.fx_graph)

        config = config["passes"]["prune"]
        config["input_generator"] = input_generator

        save_dir = root / f"mase_output/machop_test/prune/toy"
        save_dir.mkdir(parents=True, exist_ok=True)

        graph = prune_transform_pass(graph, save_dir, config)


if __name__ == "__main__":
    main()
