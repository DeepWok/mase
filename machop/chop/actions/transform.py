import os
from copy import deepcopy
from pathlib import Path
import logging

import torch
from chop.passes.graph import PASSES
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms.interface import (
    load_mase_graph_transform_pass,
    save_mase_graph_transform_pass,
)
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.checkpoint_load import load_model
from chop.tools.config_load import load_config
from chop.tools.get_input import InputGenerator, get_cf_args, get_dummy_input

logger = logging.getLogger(__name__)


def pre_transform_load(load_name: str, load_type: str, model: torch.nn.Module):
    if load_name is not None and load_type in ["pt", "pl"]:
        model = load_model(load_name=load_name, load_type=load_type, model=model)
    return model


def transform(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    task: str,
    config: str,
    save_dir: str = None,
    load_name: str = None,
    load_type: str = None,
):
    model = pre_transform_load(load_name=load_name, load_type=load_type, model=model)
    config = load_config(config)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # concrete forward args for freezing dynamic control flow in forward pass
    if "cf_args" not in config:
        cf_args = get_cf_args(model_info=model_info, task=task, model=model)
    else:
        cf_args = config["cf_args"]

    # graph generation
    graph = MaseGraph(model=model, cf_args=cf_args)
    # graph_metadata = Mase
    graph = init_metadata_analysis_pass(graph, pass_args=None)
    # logger.info(f"graph: {graph.fx_graph}")

    # create or load metadata.parameters and mase_graph.model
    if load_name is not None and load_type == "mz":
        graph = load_mase_graph_transform_pass(graph, pass_args=load_name)
    else:
        dummy_in = get_dummy_input(
            model_info=model_info,
            data_module=data_module,
            task=task,
        )
        if len(graph.model.additional_inputs) > 0:
            dummy_in = dummy_in | graph.model.additional_inputs
        graph = add_common_metadata_analysis_pass(graph, pass_args=dummy_in)
        graph = add_software_metadata_analysis_pass(graph, pass_args=None)

    pass_config = config["passes"]

    for pass_name, pass_config in pass_config.items():
        pass_name: str
        pass_config: dict
        match pass_name:
            case "quantize":
                pass_save_dir = save_dir / "quantize"
                ori_graph = deepcopy_mase_graph(graph)
                graph = PASSES["quantize"](graph, pass_args=pass_config)
                PASSES["summarize_quantization"](
                    ori_graph, graph, save_dir=pass_save_dir
                )
            case "profile_statistics":
                input_generator = InputGenerator(
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="train",
                )
                pass_config["input_generator"] = input_generator
                graph = PASSES[pass_name](graph, pass_args=pass_config)
            case "report_graph":
                pass_file_name = pass_config.get(
                    "file_name", save_dir / "report_graph.txt"
                )
                graph = PASSES[pass_name](graph, file_name=pass_file_name)
            case "report_node_type":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_meta_param":
                # {"save_path": ..., "which": "all"|["common", "hardware", "software"]}
                pass_save_path = pass_config.get("save_path", save_dir / "report")
                pass_config["save_path"] = pass_save_path
                graph = PASSES[pass_name](graph, pass_args=pass_config)
            case "report_node_shape":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_type":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_hardware_type":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_shape":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_type":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "load_mase_graph":
                pass_load_dir = pass_config["load_dir"]
                graph = PASSES[pass_name](graph, pass_args=pass_load_dir)
            case "load_node_meta_param":
                pass_load_path = pass_config["load_path"]
                graph = PASSES[pass_name](graph, pass_args=pass_load_path)
            case "save_mase_graph":
                pass_save_dir = pass_config.get(
                    "save_dir", save_dir / "saved_mase_graph"
                )
                graph = PASSES[pass_name](graph, pass_args=pass_save_dir)
            case "save_node_meta_param":
                pass_save_path = pass_config.get(
                    "save_path", save_dir / "saved_node_meta_param"
                )
                graph = PASSES[pass_name](graph, pass_args=pass_save_path)
            case "prune":
                # NOTE: The input generator is only used for when the user wants to
                # enforce or observe activation sparsity. Otherwise, it's ignored.
                # We use the validation dataloader as that doesn't shuffle the input
                # data. This determinism helps establish a fair ground in draw
                # layer-wise comparisons between activation pruning strategies.
                input_generator = InputGenerator(
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="val",
                )
                pass_config["model_name"] = model_name
                pass_config["input_generator"] = input_generator
                prune_save_dir = save_dir / "prune"
                prune_save_dir.mkdir(parents=True, exist_ok=True)
                graph = PASSES[pass_name](
                    graph,
                    save_dir=prune_save_dir,
                    config=pass_config,
                )
            case "remove_prune_wrappers":
                # Removes the pruning-related hooks and makes pruning permanent
                graph = PASSES[pass_name](graph, pass_args=None)
            case "conv_bn_fusion":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "logicnets_fusion":
                graph = PASSES[pass_name](graph, pass_args=pass_config)
            case "onnx_annotate":
                onnx_dir = save_dir / "onnx"
                onnx_dir.mkdir(parents=True, exist_ok=True)
                kwargs = {
                    "save_path": onnx_dir,
                    "data_path": pass_config["data_path"],
                }
                graph = PASSES[pass_name](graph, **kwargs)
            case _:
                my_pass = PASSES[pass_name]
                graph = my_pass(graph, pass_args=pass_config)

        assert isinstance(
            graph, MaseGraph
        ), f"Return type of {pass_name} must be MaseGraph, got {type(graph)}"

    if save_dir is not None:
        transformed_ckpt = save_dir / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        save_mase_graph_transform_pass(graph, pass_args=transformed_ckpt)
    return graph
