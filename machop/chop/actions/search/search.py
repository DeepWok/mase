import logging

import toml

from .search_space import search_space_map
from .strategies import strategy_map

from chop.passes.graph.mase_graph import MaseGraph
from chop.passes import init_metadata_analysis_pass, add_common_metadata_analysis_pass
from chop.tools.get_input import get_dummy_input
from chop.models import nlp_models

from chop.models.manual.opt_quantized.configuration_opt import OPTQuantizedConfig

logger = logging.getLogger(__name__)


def parse_search_config(search_config):
    with open(search_config, "r") as f:
        search_args = toml.load(f)
    # building search space
    strategy_config = search_args["strategy"]
    search_space_config = search_args["search_space"]

    assert strategy_config["data_loader"] in [
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
    ]
    return (strategy_config, search_space_config)


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
    # search prep
    logger.info("Search started...")
    configs = parse_search_config(search_config)
    strategy_config, search_space_config = configs
    name = search_space_config.get("style", None)

    if name is None or not (name in search_space_map):
        possible_names = list(search_space_map.keys())
        raise ValueError(f"{name} must be defined in {possible_names}.")

    runner = search_space_config.get("runner", "graph")
    # construct a minimal mase graph
    if runner in ["graph", "mg", "mase_graph"]:
        # TODO: it seems like get_dummy_input may fail on wikitext2
        # TODO: mase graph based runners are not refactored yet
        is_nlp_model = model_name in nlp_models
        dummy_input = get_dummy_input(data_module, task, is_nlp_model)
        mg = MaseGraph(model)
        mg = init_metadata_analysis_pass(mg, None)
        mg = add_common_metadata_analysis_pass(mg, dummy_input)
    elif runner in ["module"]:
        mg = None
    else:
        raise ValueError(f"runner {runner} is not supported.")

    # construct a search space
    search_space_cls = search_space_map[name]
    search_space = search_space_cls(
        model_name=model_name,
        model=model,
        mg=mg,
        config=search_space_config,
        accelerator=accelerator,
    )
    search_space.build_search_space()

    # construct a search strategy
    name = strategy_config.get("name", None)
    strategy_cls = strategy_map[name]

    strategy = strategy_cls(
        model_name=model_name,
        model=model,
        mg=mg,
        task=task,
        info=info,
        data_module=data_module,
        accelerator=accelerator,
        config=strategy_config,
        save_dir=save_path,
    )

    best_metric, best_sample, best_model = strategy.search(search_space)
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
