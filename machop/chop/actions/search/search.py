import copy
import logging
import operator
import os
import random
from functools import partial

import optuna
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F

from .search_space import search_space_map
from .strategies import strategy_map
from .runner import runner_map

from chop.passes.graph.mase_graph import MaseGraph
from chop.passes import init_metadata_analysis_pass, add_mase_ops_analysis_pass


logger = logging.getLogger(__name__)


def parse_search_config(search_config):
    with open(search_config, "r") as f:
        search_args = toml.load(f)
    # building search space
    strategy_config = search_args["strategy"]

    search_space_config = search_args["search_space"]

    search_runner_config = search_args["runner"]
    assert search_runner_config["data_loader"] in [
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
    ]
    return (strategy_config, search_space_config, search_runner_config)


def search(
    model_name,
    model,
    task,
    info,
    data_module,
    search_config,
    save_path,
    accelerator,
    load_name,
    load_type,
):
    logger.info("Search started...")
    strategy_config, search_space_config, runner_config = parse_search_config(
        search_config
    )

    name = search_space_config.get("name", None)
    if name is None or not (name in search_space_map):
        possible_names = list(search_space_map.keys())
        raise ValueError(f"{name} must be defined in {possible_names}.")

    # construct a minimal mase graph
    mg = MaseGraph(model)
    mg = init_metadata_analysis_pass(mg, None)
    mg = add_mase_ops_analysis_pass(mg)

    # construct a search space
    search_space_cls = search_space_map[name]
    search_space = search_space_cls(
        model_name=model_name, model=model, mg=mg, config=search_space_config
    )
    search_space.build_search_space()

    # construct a search strategy
    name = strategy_config.get("name", None)
    strategy_cls = strategy_map[name]
    strategy = strategy_cls(strategy_config)

    # construct a search runner
    name = runner_config.get("name", None)
    runner = runner_map[name](
        model_name,
        model,
        mg,
        task,
        info,
        data_module,
        accelerator,
        runner_config,
        save_path,
    )
    best_metric, best_sample, best_model = strategy.search(search_space, runner)
    print(best_metric, best_sample)

    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    # searcher = SearchQuantization(
    #     model_name=model_name,
    #     model=model,
    #     is_nlp_model=is_nlp_model,
    #     task=task,
    #     info=info,
    #     modifier_kwargs=modifier_kwargs,
    #     data_module=data_module,
    #     search_config=search_config,
    #     save_dir=save_dir,
    #     accelerator=accelerator,
    # )
    # searcher.search()
    # searcher.save_study_and_config()
    # logger.info("Search finished.")
