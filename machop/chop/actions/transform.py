import os

import torch
from chop.passes import passes
from chop.passes.analysis import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    report_metadata_analysis_pass,
)
from chop.passes.graph.mase_graph import MaseGraph
from chop.passes.transforms.interface import (
    load_mase_graph_transform_pass,
    save_mase_graph_transform_pass,
)
from chop.tools.checkpoint_load import load_model
from chop.tools.config_load import load_config
from chop.tools.get_input import get_cf_args


def pre_transform_load(load_name: str, load_type: str, model: torch.nn.Module):
    if load_name is not None and load_type in ["pt", "pl"]:
        model = load_model(load_name=load_name, laod_type=load_type, model=model)
    return model


def transform(
    model_name: str,
    model: torch.nn.Module,
    info: dict,
    task: str,
    data_module,
    config: str,
    save_dir: str = None,
    load_name: str = None,
    load_type: str = None,
):
    model = pre_transform_load(load_name=load_name, load_type=load_type, model=model)
    config = load_config(config)
    # concrete forward args for freezing dynamic control flow in forward pass
    if "cf_args" not in config:
        cf_args = get_cf_args(model_name=model_name, task=task, model=model)
    else:
        cf_args = config["cf_args"]

    # graph generation
    graph = MaseGraph(model=model, cf_args=cf_args)
    # graph_metadata = Mase
    graph = init_metadata_analysis_pass(graph, pass_args=None)

    # create or load metadata.parameters and mase_graph.model
    if load_name is not None and load_type == "mz":
        graph = load_mase_graph_transform_pass(graph, pass_args=load_name)
    else:
        graph = add_common_metadata_analysis_pass(graph, pass_args=None)
    graph = report_metadata_analysis_pass(graph, pass_args=None)

    # passes
    pass_config = config["passes"]
    for pass_name, pass_config in pass_config.items():
        my_pass = passes[pass_name]
        graph = my_pass(graph, pass_args=pass_config)

    # save transformed model
    if save_dir is not None:
        save_mase_graph_transform_pass(graph, pass_args=save_dir)
    return graph
