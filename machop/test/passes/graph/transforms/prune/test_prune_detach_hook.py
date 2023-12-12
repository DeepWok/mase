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
sys.path.append(Path(__file__).resolve().parents[5].as_posix())

import chop.models as models
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    profile_statistics_analysis_pass,
    prune_transform_pass,
)

from chop.passes.graph.analysis.pruning.calculate_sparsity import (
    add_pruning_metadata_analysis_pass,
)

from chop.passes.graph.transforms.pruning.prune_detach_hook import (
    prune_detach_hook_transform_pass,
)

from chop.ir.graph.mase_graph import MaseGraph
from chop.tools.get_input import InputGenerator, get_dummy_input
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity
import pprint

set_logging_verbosity("debug")

logger = logging.getLogger("chop.test")
pp = pprint.PrettyPrinter(indent=4)

configs = ["scope_local_granularity_elementwise_method_random"]


def test_prune_detach_hook():
    for c in configs:
        run_with_config(c)


def run_with_config(config_file):
    BATCH_SIZE = 32

    root = Path(__file__).resolve().parents[5]
    config_file = root / f"configs/tests/prune/{config_file}.toml"
    with open(config_file) as f:
        config = toml.load(f)

    model_name = "vgg7"
    dataset_name = "cifar10"

    # NOTE: We're only concerned with pre-trained vision models
    dataset_info = get_dataset_info(dataset_name)
    model_info = models.get_model_info(model_name)
    data_module = MaseDataModule(
        model_name=model_name,
        name=dataset_name,
        batch_size=BATCH_SIZE,
        num_workers=0,
        tokenizer=None,
        max_token_len=None,
    )
    data_module.prepare_data()
    data_module.setup()
    # NOTE: We only support vision classification models for now.
    dummy_input = get_dummy_input(model_info, data_module, "cls", "cpu")

    # We need the input generator to do a sample forward pass to log information on
    # the channel-wise activation sparsity.
    input_generator = InputGenerator(
        model_info=model_info,
        data_module=data_module,
        task="cls",
        which_dataloader="train",
    )

    model = models.get_model(model_name, "cls", dataset_info, pretrained=True)

    _ = model(dummy_input["x"])
    graph = MaseGraph(model=model)

    # NOTE: Both functions have pass arguments that are not used in this example
    graph, _ = init_metadata_analysis_pass(graph, None)
    graph, _ = add_common_metadata_analysis_pass(
        graph,
        {
            "dummy_in": dummy_input,
            # set add_value to True, because activation pruning makes use of real activation values
            "add_value": True,
            "force_device_meta": False,
        },
    )
    graph, _ = add_software_metadata_analysis_pass(graph, None)

    profile_pass_arg = {
        "by": "type",
        "target_weight_nodes": [
            "conv2d",
        ],
        "target_activation_nodes": [
            "conv2d",
        ],
        "weight_statistics": {
            "variance_precise": {"device": "cpu", "dims": "all"},
        },
        "activation_statistics": {
            "variance_precise": {"device": "cpu", "dims": "all"},
        },
        "input_generator": input_generator,
        "num_samples": 1,
    }

    graph, _ = profile_statistics_analysis_pass(graph, profile_pass_arg)

    config = config["passes"]["prune"]
    config["input_generator"] = input_generator
    config["dummy_in"] = dummy_input

    # save_dir = root / f"mase_output/machop_test/prune/{config_name}"
    # save_dir.mkdir(parents=True, exist_ok=True)

    # The default save directory is specified as the current working directory
    graph, _ = prune_transform_pass(graph, config)
    graph, sparsity_info = add_pruning_metadata_analysis_pass(
        graph, {"dummy_in": dummy_input, "add_value": False}
    )
    graph, _ = prune_detach_hook_transform_pass(graph, {})

    pp.pprint(sparsity_info)


test_prune_detach_hook()
