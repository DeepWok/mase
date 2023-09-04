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
    prune_transform_pass,
    add_software_metadata_analysis_pass,
)
from chop.passes.graph.mase_graph import MaseGraph
from chop.tools.logger import getLogger
from chop.tools.get_input import InputGenerator, get_dummy_input
from chop.dataset import MaseDataModule, get_dataset_info

logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    # We don't want to run this script if there's no path provided
    if len(sys.argv) < 2:
        return

    config_name = sys.argv[1]  # Name of config to load from configs/tests/prune
    BATCH_SIZE = 32

    root = Path(__file__).resolve().parents[5]
    config_path = root / f"machop/configs/tests/prune/{config_name}.toml"
    with open(config_path) as f:
        config = toml.load(f)

        # NOTE: We're only concerned with pre-trained vision models
        dataset_info = get_dataset_info(config["dataset"])
        model_info = models.get_model_info(config["model"])
        data_module = MaseDataModule(
            model_name=config["model"],
            name=config["dataset"],
            batch_size=BATCH_SIZE,
            num_workers=os.cpu_count(),
            tokenizer=None,
            max_token_len=None,
        )
        data_module.prepare_data()
        data_module.setup()
        # NOTE: We only support vision classification models for now.
        dummy_input = get_dummy_input(data_module, "cls", is_nlp_model=False)

        # We need the input generator to do a sample forward pass to log information on
        # the channel-wise activation sparsity.
        input_generator = InputGenerator(
            model_info=model_info,
            data_module=data_module,
            task="cls",
            which_dataloader="train",
        )

        model_inst_fn = models.get_model(config["model"], task="cls")
        model = model_inst_fn(dataset_info, pretrained=True)

        graph = MaseGraph(model=model)
        # NOTE: Both functions have pass arguments that are not used in this example
        graph = init_metadata_analysis_pass(graph, None)
        graph = add_common_metadata_analysis_pass(graph, dummy_input)
        graph = add_software_metadata_analysis_pass(graph, None)
        logger.debug(graph.fx_graph)

        config = config["passes"]["prune"]
        config["input_generator"] = input_generator

        save_dir = root / f"mase_output/machop_test/prune/{config_name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # The default save directory is specified as the current working directory
        graph = prune_transform_pass(graph, save_dir, config)


if __name__ == "__main__":
    main()
