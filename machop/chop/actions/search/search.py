import logging
from os import PathLike

import toml
import torch
from fvcore.common.config import CfgNode
from ...tools.checkpoint_load import load_model
from ...tools.config_load import load_config
from ...tools.get_input import get_dummy_input
from .search_space import get_search_space_cls
from .strategies import get_search_strategy_cls
from chop.tools.utils import device
from chop.tools.utils import parse_accelerator


# For nas proxy search
import numpy as np
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_zc_benchmark_api,get_dataset_api
from naslib.utils import get_train_val_loaders, get_project_root
from naslib.predictors import ZeroCost
from naslib.search_spaces.core import Metric

# For training  meta-proxy network
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from einops import rearrange
from torch import optim
import torch.nn.functional as F




logger = logging.getLogger(__name__)
### Function made for nas config

    
def parse_search_config(config):
    """
    Parse search config from a dict or a toml file and do sanity check.

    ---
    The search config must consist of two parts: strategy and search_space.
    """        

    if not isinstance(config, dict):
        search_config = load_config(config)

    search_config = config["search"]  # the actual config for action search
    strategy_config = search_config["strategy"]
    search_space_config = search_config["search_space"]
    # print(search_config.keys())
    if strategy_config['setup']['n_trials'] < 1:
        logger.info('Invalid Number of Trials!')
        exit()
    return strategy_config, search_space_config


def search(
    model: torch.nn.Module,
    model_info,
    task: str,
    dataset_info,
    data_module,
    search_config: dict | PathLike,
    save_path: PathLike,
    accelerator: str,
    load_name: PathLike = None,
    load_type: str = None,
    visualizer=None,
):
    """
    Args:
        model: the model to be searched
    """
    # search preparation
    accelerator = parse_accelerator(accelerator)
    #######################################################
    ### Our contribution here:
    if not isinstance(search_config, dict):
        search_config = load_config(search_config)
    searchconfig=search_config['search']
    
    ### End of our contribution
    strategy_config, search_space_config = parse_search_config(search_config)
    save_path.mkdir(parents=True, exist_ok=True)

    # load model if the save_name is provided
    if load_name is not None and load_type in ["pl", "mz", "pt"]:
        model = load_model(load_name=load_name, load_type=load_type, model=model)
        logger.info(f"Loaded model from {load_name}.")
    model.to(accelerator)
    # set up data module
    data_module.prepare_data()
    data_module.setup()
    
    # construct the search space
    logger.info("Building search space...")
    search_space_cls = get_search_space_cls(search_space_config["name"])
    search_space = search_space_cls(
        model=model,
        model_info=model_info,
        config=search_space_config,
        dummy_input=get_dummy_input(model_info, data_module, task, device=accelerator),
        accelerator=accelerator,
    )
    search_space.build_search_space()
    dummy_input = get_dummy_input(model_info, data_module, task, device=accelerator)

    strategy_cls = get_search_strategy_cls(strategy_config["name"])

    strategy = strategy_cls(
        model_info=model_info,
        task=task,
        dataset_info=dataset_info,
        data_module=data_module,
        config=strategy_config,
        accelerator=accelerator,
        save_dir=save_path,
        visualizer=visualizer,
    )

    logger.info("Search started...")
    # perform search and save the results
    strategy.search(search_space)
