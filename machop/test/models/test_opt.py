# This example converts a simple MLP model to Verilog
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[3].joinpath("machop").as_posix())

from chop.ir import MaseGraph
from chop.models import get_model, get_model_info, get_tokenizer
from chop.tools import get_cf_args, get_dummy_input, load_config
from chop.dataset import get_dataset_info, MaseDataModule
import chop.passes as passes


def test_opt():
    machop_dir = Path(__file__).resolve().parents[3] / "machop"
    config_toml = machop_dir / "configs" / "tests" / "quantize" / "fixed.toml"
    assert config_toml.exists(), f"config_toml {config_toml} does not exist"
    config = load_config(config_toml)

    load_pretrained = True

    # Get model
    wikitext_info = get_dataset_info("wikitext2")
    opt = get_model(
        "facebook/opt-125m:patched",
        task="lm",
        dataset_info=wikitext_info,
        pretrained=True,
    )
    opt_tokenizer = get_tokenizer("facebook/opt-125m:patched")

    print(f"prepare data")
    # Get data module for dummy inputs
    data_module = MaseDataModule(
        name="wikitext2",
        batch_size=1,
        num_workers=os.cpu_count(),
        max_token_len=128,
        tokenizer=opt_tokenizer,
        load_from_cache_file=True,
        model_name="facebook/opt-125m@patched",
    )
    data_module.prepare_data()
    data_module.setup()

    model_info = get_model_info("facebook/opt-125m:patched")
    cf_args = get_cf_args(model_info=model_info, task="lm", model=opt)

    graph = MaseGraph(model=opt, cf_args=cf_args)

    # Generate dummy input
    dummy_in = get_dummy_input(model_info, data_module=data_module, task="lm")
    if len(graph.model.additional_inputs) > 0:
        dummy_in = dummy_in | graph.model.additional_inputs

    # Generate graph and initialize metadata
    print(f"init metadata")
    graph, _ = passes.init_metadata_analysis_pass(graph, pass_args=None)
    # graph, _ = passes.add_common_metadata_analysis_pass(
    #     graph, pass_args={"dummy_in": dummy_in}
    # )

    # Quantize and add hardware metadata
    # graph, _ = passes.quantize_transform_pass(graph, config["passes"]["quantize"])
    # graph, _ = passes.add_hardware_metadata_analysis_pass(graph, pass_args=None)

    # ==============================================
    # TO DO: TEMPORARY
    # print(f"before removal {graph.nodes_in}")
    # graph.nodes_in = graph.nodes_in[:-3]
    # print(f"after removal {graph.nodes_in}")

    # print(f"UPDATE ATTENTION ARGS")
    # Rename attention node inputs etc
    # for node in graph.fx_graph.nodes:
    #     if "self_attn" in node.name and "layer_norm" not in node.name:
    #         node.meta["mase"].parameters["common"]["args"]["bias_q"] = (
    #             node.meta["mase"].parameters["common"]["args"].pop("q_proj.bias")
    #         )
    #         node.meta["mase"].parameters["common"]["args"]["bias_k"] = (
    #             node.meta["mase"].parameters["common"]["args"].pop("k_proj.bias")
    #         )
    #         node.meta["mase"].parameters["common"]["args"]["bias_v"] = (
    #             node.meta["mase"].parameters["common"]["args"].pop("v_proj.bias")
    #         )

    #         node.meta["mase"].parameters["common"]["args"]["weight_q"] = (
    #             node.meta["mase"].parameters["common"]["args"].pop("q_proj.weight")
    #         )
    #         node.meta["mase"].parameters["common"]["args"]["weight_k"] = (
    #             node.meta["mase"].parameters["common"]["args"].pop("k_proj.weight")
    #         )
    #         node.meta["mase"].parameters["common"]["args"]["weight_v"] = (
    #             node.meta["mase"].parameters["common"]["args"].pop("v_proj.weight")
    #         )

    #         # Pop out attention_mask and output_attentions
    #         node.meta["mase"].parameters["common"]["args"].pop("data_in_2")
    #         node.meta["mase"].parameters["common"]["args"].pop("data_in_4")

    #         # Pop output projection weight/bias
    #         node.meta["mase"].parameters["common"]["args"].pop("out_proj.weight")
    #         node.meta["mase"].parameters["common"]["args"].pop("out_proj.bias")

    # ==============================================

    # graph, _ = passes.report_node_type_analysis_pass(graph)

    # Emit verilog
    # graph, _ = passes.emit_verilog_top_transform_pass(graph)


if __name__ == "__main__":
    test_opt()
