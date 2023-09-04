#!/usr/bin/env python3
# This example converts a toy model to LUTNet combining pruning and training steps. ONLY FOR TESTING PURPOSES.
import logging
import os
import sys
from pathlib import Path

import toml

import torch
import torch.nn as nn

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[4].as_posix())

from chop.passes import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    prune_transform_pass,
    add_software_metadata_analysis_pass,
    verify_common_metadata_analysis_pass,  # TODO: verification verify_common_metadata_analysis_pass does not work
)

from chop.passes.graph.mase_graph import MaseGraph
from chop.passes.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.passes.utils import deepcopy_mase_graph
from chop.tools.logger import getLogger
from chop.models import get_model_info

# pruning
from chop.tools.logger import getLogger
from chop.dataset import MaseDataModule
from chop.tools.get_input import InputGenerator, get_dummy_input


logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


# --------------------------------------------------
#   Model specifications
# --------------------------------------------------
class ToyNet(nn.Module):
    def __init__(self, image_size, num_classes):
        super(ToyNet, self).__init__()
        in_planes = image_size[0] * image_size[1] * image_size[2]
        self.seq_blocks = nn.Sequential(
            nn.Linear(in_planes, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.seq_blocks(x.view(x.size(0), -1))


# --------------------------------------------------
#   Emit Verilog using Mase
# --------------------------------------------------
def main():
    BATCH_SIZE = 32

    root = Path(__file__).resolve().parents[5]
    pruning_config_path = root / "machop/configs/tests/prune/lutnet.toml"
    with open(pruning_config_path, "r") as f:
        config = toml.load(f)

        data_module = MaseDataModule(
            model_name="toy_tiny",  # This doesn't really matter
            name="cifar10",
            batch_size=BATCH_SIZE,
            num_workers=os.cpu_count(),
            tokenizer=None,
            max_token_len=None,
        )
        data_module.prepare_data()
        data_module.setup()

        # NOTE: We only support vision classification models for now.

        model_info = get_model_info("toy_tiny")
        dummy_input = get_dummy_input(model_info, data_module, "cls")

        # We need the input generator to do a sample forward pass to log information on
        # the channel-wise activation sparsity.
        input_generator = InputGenerator(
            model_info=model_info,
            data_module=data_module,
            task="cls",
            which_dataloader="train",
        )

        model = ToyNet(image_size=[3, 32, 32], num_classes=10)
        graph = MaseGraph(model=model)

        # NOTE: First round training

        graph = init_metadata_analysis_pass(graph, None)
        graph = add_common_metadata_analysis_pass(graph, dummy_input)
        graph = add_software_metadata_analysis_pass(graph, None)
        logger.debug(graph.fx_graph)

        config = config["passes"]["prune"]
        config["input_generator"] = input_generator

        save_dir = root / f"mase_output/machop_test/prune/toy"
        save_dir.mkdir(parents=True, exist_ok=True)

        graph = prune_transform_pass(graph, save_dir, config)

        # binary_config = toml.load(f1)
        binary_config = {
            "by": "type",
            "report": True,
            "default": {"config": {"name": None}},
            "linear": {
                "config": {
                    "name": "binary",
                    "binary_training": "NA",
                    "data_in_width": 32,
                    "data_in_frac_width": 16,
                    "data_in_stochastic": True,
                    "data_in_bipolar": True,
                    "weight_width": 1,
                    "weight_stochastic": True,
                    "weight_bipolar": True,
                    "bias_width": 1,
                    "bias_stochastic": True,
                    "bias_bipolar": True,
                }
            },
        }
        # graph = verify_common_metadata_analysis_pass(graph)
        ori_graph = deepcopy_mase_graph(graph)
        graph = quantize_transform_pass(graph, binary_config)

        summarize_quantization_analysis_pass(
            ori_graph, graph, save_dir="quantize_summary"
        )

        # NOTE: LUTNet initialization
        # k (int): Number of inputs for each table.
        # binarization_level (int): which level of binarization is applied, 0 no binarization , "binarized_weight" is only weights binarized.
        # input_expanded (bool): If set to True, means all LUT's inputs are considered during calculations , else only the first input will considered and the remaining will be masked.

        lutnet_quantize_config = {
            "by": "type",
            "baseline_weight_path": "/workspace/mase_output/toy_classification_cifar10_2023-08-21/software/transform/transformed_ckpt_bl/transformed_ckpt/graph_module.mz",
            "default": {"config": {"name": None}},
            "linear": {
                "config": {
                    "name": "lutnet",
                    # data
                    "data_in_k": 2,
                    "data_in_input_expanded": True,
                    "data_in_binarization_level": "binarized_weight",
                    "data_in_width": 8,
                    "data_in_frac_width": 4,
                    "data_in_dim": None,
                    # weight
                    "weight_width": 8,
                    "weight_frac_width": 4,
                    "weight_k": None,
                    "weight_input_expanded": None,
                    "weight_binarization_level": None,
                    "weight_in_dim": None,
                    # bias
                    "bias_width": 8,
                    "bias_frac_width": 4,
                    "bias_k": None,
                    "bias_input_expanded": None,
                    "bias_binarization_level": None,
                    "bias_in_dim": None,
                }
            },
        }
        # graph = verify_common_metadata_analysis_pass(graph)
        binary_graph = deepcopy_mase_graph(graph)
        # NOTE: A proper baseline checkpoint is needed to run this transform. Specify the baseline checkpoint in baseline_weight_path
        # graph = quantize_transform_pass(graph, lutnet_quantize_config)

        summarize_quantization_analysis_pass(
            binary_graph, graph, save_dir="lutnet_quantize_summary"
        )

        # mg = report(mg)
        # mg = emit_verilog(mg)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
if __name__ == "__main__":
    main()
