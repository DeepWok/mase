import logging
from os import PathLike

import pandas as pd
import toml
import torch

from ...tools.checkpoint_load import load_model
from ...tools.config_load import load_config
from ...tools.get_input import get_dummy_input
from .search_space import get_search_space_cls
from .strategies import get_search_strategy_cls
from chop.tools.utils import device
from chop.tools.utils import parse_accelerator

logger = logging.getLogger(__name__)


def parse_search_config(search_config):
    """
    Parse search config from a dict or a toml file and do sanity check.

    ---
    The search config must consist of two parts: strategy and search_space.
    """
    if not isinstance(search_config, dict):
        search_config = load_config(search_config)
    search_config = search_config["search"]  # the actual config for action search
    strategy_config = search_config["strategy"]
    search_space_config = search_config["search_space"]
    print(strategy_config)
    print(search_space_config)

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
    # print("model info")
    # print(model_info)
    # print("data_module")
    # print(data_module)

    accelerator = parse_accelerator(accelerator)
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

    # construct a search strategy
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

    
    '''
    group 2: zero cost

    This part of the code is responsible for executing the zero cost strategy if it's enabled. 

    - First, it checks if the zero cost mode is enabled.
    - If enabled, it calculates the weights for the strategy.
    - The proxy logger from the strategy is then retrieved and converted to a string for logging.
    - The proxy logger is then logged.
    - The accuracy is predicted using the model and the proxy logger.
    - A DataFrame is created to store the results, which includes the index of the architecture, the predicted accuracy, and the true accuracy.
    - The results are then sorted by the predicted accuracy in descending order and logged.
    - Finally, the proxy logger and the sorted results are saved to Excel files.
    '''
    if strategy.zero_cost_mode:
        strategy.zero_cost_weight()

        # logger.info("model coefficients: ", strategy.zc_weight_model.coef_)
        # logger.info("model intercept: ", strategy.zc_weight_model.intercept_)
        
        # print("values of proxies", strategy.zc_proxy)
        proxy_logger = strategy.zc_proxy
        proxy_logger_str = proxy_logger.to_string()
        logger.info("Proxy Logger:\n%s", proxy_logger_str)

        # strategy.zc_true_accuracy.to_excel("../results/accuracy1.xlsx")
        predicted_accuracy = strategy.zc_weight_model.predict(strategy.zc_proxy)
        
        results = pd.DataFrame({
            'Architecture_Index': range(len(predicted_accuracy)),
            'Predicted_Accuracy': predicted_accuracy,
            'True Accuracy': strategy.zc_true_accuracy
        })
        
        sorted_results = results.sort_values(by='Predicted_Accuracy', ascending=False)
        logger.info("Sorted Results:\n%s", sorted_results)
        
        strategy.zc_proxy.to_excel("/home/xz2723/mase_xinyi/machop/results/proxy_4.xlsx")
        sorted_results.to_excel("/home/xz2723/mase_xinyi/machop/results/sorted_results_4.xlsx")
        