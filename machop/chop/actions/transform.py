import torch
import os

from chop.models import nlp_models
from chop.tools import load_model
from chop.tools.get_input import get_dummy_inputs
from chop.passes.graph.mase_graph import MaseGraph

from chop.passes.analysis import add_common_metadata_analysis_pass, init_metadata_analysis_pass
from chop.passes.transforms import quantize_transform_pass
from chop.passes import passes
from chop.tools import load_config



def transform(
    model_name,
    model,
    info,
    task,
    data_module,
    config,
    save_path,
    load_name,
    load_type,
):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    model_load_candidate = model["model"] if model in nlp_models else model
    if load_name is not None:
        model["model"] = load_model(
            load_name=load_name,
            load_type=load_type,
            model=model_load_candidate)

    model = model["model"] if model in nlp_models else model

    dummy_inputs = get_dummy_inputs(
        model_name=model_name,
        task=task,
        model=model)
    
    custom_kwargs = {}

    kwargs = {
        "model": model,
        "config_path": config,
        "dummy_inputs_for_fix": dummy_inputs,
        "save_dir": os.path.join(save_path, "transformed_model"),
    }

    kwargs |= custom_kwargs

    pass_config = load_config(config)['passes']

    # graph generation
    graph = MaseGraph(model=model, cf_args=dummy_inputs)
    # graph_metadata = Mase
    graph = init_metadata_analysis_pass(graph, None)
    graph = add_common_metadata_analysis_pass(graph, None)

    for pass_name, pass_config in pass_config.items():
        my_pass = passes[pass_name]
        graph = my_pass(graph, pass_args=pass_config)
