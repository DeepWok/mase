# This example converts a simple MLP model to Verilog
import os, sys, logging

import torch
import torch.nn as nn

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "machop",
    )
)
from chop.ir.graph.mase_graph import MaseGraph

from chop.passes.graph.analysis import (
    add_hardware_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    verify_common_metadata_analysis_pass,
    report_node_type_analysis_pass,
    report_node_shape_analysis_pass,
    report_node_hardware_type_analysis_pass,
)
from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_mlir_hls_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_logicnets_transform_pass,
    emit_bram_transform_pass,
    emit_verilog_tb_transform_pass,
    quantize_transform_pass,
)

# Needed for loading MaseGraph from checkpoint
from chop.passes.graph.transforms.interface import load_mase_graph_transform_pass
from chop.actions.transform import pre_transform_load
from chop.tools.config_load import (
    post_parse_load_config,
    load_config,
)
from chop.tools.get_input import get_cf_args

from chop.models import (
    get_model,
    get_model_info,
)

from chop.dataset import get_dataset_info

# from chop.tools.logger import get_logger

# logger = get_logger("chop")
# logger.setLevel(logging.DEBUG)


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
class MLP(torch.nn.Module):
    """
    Toy model for jet substructure classifcation (from LogicNets paper)
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 5)
        # self.fc3 = nn.Linear(28 * 28 * 4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.nn.functional.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


def test_emit_top_logicnets_verilog():
    # -------- MaseGraph from model -------- #
    # mlp = MLP()
    # mg = MaseGraph(model=mlp)

    # -------- MaseGraph from checkpoint -------- #
    load_name = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "mase_output",
        "jsc-toy_classification_jsc_2023-10-04",
        "software",
        "transform",
        "transformed_ckpt",
        "graph_module.mz",
    )
    load_type = "mz"
    model_name = "jsc-s"
    task = "physical"
    dataset_name = "jsc"

    dataset_info = get_dataset_info(dataset_name)
    model_info = get_model_info(model_name)
    model = get_model(
        name=model_name,
        task=task,
        dataset_info=dataset_info,
        checkpoint=load_name,
        pretrained=True,
    )
    model = pre_transform_load(load_name=load_name, load_type=load_type, model=model)
    config = {}

    # concrete forward args for freezing dynamic control flow in forward pass
    if "cf_args" not in config:
        cf_args = get_cf_args(model_info=model_info, task=task, model=model)
    else:
        cf_args = config["cf_args"]

    # graph generation
    mg = MaseGraph(model=model, cf_args=cf_args)
    mg = init_metadata_analysis_pass(mg, pass_args=None)
    mg = load_mase_graph_transform_pass(mg, pass_args=load_name)

    # logger.debug(f"graph: {mg.fx_graph}")

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 1, 16))
    dummy_in = {"x": x}

    mg = init_metadata_analysis_pass(mg, None)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)
    # mg = report_node_shape_analysis_pass(mg)

    # Sanity check and report - verify or compare with expected results here
    # mg = verify_common_metadata_analysis_pass(mg)

    # # Quantize to int
    # config_file = os.path.join(
    #     os.path.dirname(os.path.realpath(__file__)),
    #     "..",
    #     "..",
    #     "..",
    #     "..",
    #     "configs",
    #     "logicnets",
    #     "integer_logicnets.toml",
    # )
    # mg = report_node_type_analysis_pass(mg)

    # # load toml config file
    # with open(config_file, "r") as f:
    #     quan_args = toml.load(f)["passes"]["quantize"]
    # mg = quantize_transform_pass(mg, quan_args)

    # There is a bug in the current quantization pass, where the metadata is not updated with the precision.
    # Here we temporarily update the metadata here so we can test the hardware back end.
    for node in mg.fx_graph.nodes:
        for arg, _ in node.meta["mase"].parameters["common"]["args"].items():
            node.meta["mase"].parameters["common"]["args"][arg]["type"] = "fixed"
            node.meta["mase"].parameters["common"]["args"][arg]["precision"] = [8, 3]
        for result, _ in node.meta["mase"].parameters["common"]["results"].items():
            node.meta["mase"].parameters["common"]["results"][result]["type"] = "fixed"
            node.meta["mase"].parameters["common"]["results"][result]["precision"] = [
                8,
                3,
            ]

    mg = report_node_type_analysis_pass(mg)
    mg = add_hardware_metadata_analysis_pass(mg)
    mg = report_node_hardware_type_analysis_pass(mg)
    # mg = verify_hardware_metadata_analysis_pass(mg)

    # mg = emit_verilog_top_transform_pass(mg)
    # mg = emit_logicnets_transform_pass(mg)
    # mg = emit_bram_transform_pass(mg)
    mg = emit_internal_rtl_transform_pass(mg)
    print("Done")
    # # For internal models, the test inputs can be directly fetched from the dataset
    # # using InputGenerator from chop.tools.get_input
    # cosim_config = {"test_inputs": [x], "trans_num": 1}
    # mg = emit_verilog_tb_transform_pass(mg, pass_args=cosim_config)
