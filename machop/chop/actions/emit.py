import torch.nn
import os
from os import PathLike
from ..tools.checkpoint_load import load_model

from ..ir.graph.mase_graph import MaseGraph

from ..passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)

from ..passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

from ..tools.get_input import InputGenerator, get_cf_args, get_dummy_input
from ..tools.config_load import load_config

from pathlib import Path


def emit(
    model: torch.nn.Module,
    model_info,
    task: str,
    dataset_info,
    data_module,
    load_name: PathLike = None,
    load_type: str = None,
):
    mg = MaseGraph(model=model)
    mg, _ = init_metadata_analysis_pass(mg, None)

    # data_module.prepare_data()
    # data_module.setup()
    # dummy_in = get_dummy_input(
    #     model_info=model_info,
    #     data_module=data_module,
    #     task=task,
    #     device="cpu",
    # )
    dummy_in = {"x": torch.Tensor([[[[-0.2368, 0.4142], [0.6548, 0.6421]]]])}
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )

    # Temporary: quantizing the graph since mase only has fixed precision hardware components
    # In the future, either expect a pre-quantized mz checkpoint, or emit floating point components
    config_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "configs",
        "tests",
        "quantize",
        "fixed.toml",
    )
    quan_args = load_config(config_file)["passes"]["quantize"]
    mg, _ = quantize_transform_pass(mg, quan_args)

    # TEMPORARY: Update the metadata (see https://github.com/jianyicheng/mase-tools/issues/502)
    for node in mg.fx_graph.nodes:
        for arg, arg_info in node.meta["mase"]["common"]["args"].items():
            if isinstance(arg_info, dict):
                arg_info["type"] = "fixed"
                arg_info["precision"] = [8, 3]
        for result, result_info in node.meta["mase"]["common"]["results"].items():
            if isinstance(result_info, dict):
                result_info["type"] = "fixed"
                result_info["precision"] = [8, 3]

    mg, _ = add_hardware_metadata_analysis_pass(mg, None)

    # Now the fun stuff...
    mg, _ = emit_verilog_top_transform_pass(mg)
    mg, _ = emit_internal_rtl_transform_pass(mg)
    mg, _ = emit_bram_transform_pass(mg)
    mg, _ = emit_cocotb_transform_pass(mg)
