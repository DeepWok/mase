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
from chop.NASLib.naslib.search_spaces import NasBench201SearchSpace
from chop.NASLib.naslib.utils import get_zc_benchmark_api,get_dataset_api
from chop.NASLib.naslib.utils import get_train_val_loaders, get_project_root
from chop.NASLib.naslib.predictors import ZeroCost
from chop.NASLib.naslib.search_spaces.core import Metric


logger = logging.getLogger(__name__)
### Function made for nas config
def parse_nas_config(config):
    search_config = config["search"]   #p arse into search config
    nas_config = search_config['nas']  # parse into nas
    op_config = nas_config['op_config']
    proxy_config = nas_config['proxy_config']
    return op_config, proxy_config
    

    
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
    
    ### Our contribution here:
    if isinstance(search_config['search']['nas'],dict):
        op_config, proxy_config = parse_nas_config(search_config)   # type(op_config) = list of tuple , type(proxy_config) = list of strings
        # Prepare dict for recording scores
        scores = {}
        for proxy_name in proxy_config:
            scores[proxy_name] = []
        train_accuries = []
        val_accuries = []

        for op in op_config:
            # config_dict is config from nas-bench
            config_dict = {
                'dataset': 'cifar10', # Dataset to loader: can be cifar10, cifar100, ImageNet16-120
                'data': str(get_project_root()) + '/data', # path to naslib/data where cifar is saved
                'search': {
                    'seed': 9001, # Seed to use in the train, validation and test dataloaders
                    'train_portion': 0.7, # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
                    'batch_size': 32, # batch size of the dataloaders
                }
            }
            config = CfgNode(config_dict)
            train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)

            # Generate models
            graph = NasBench201SearchSpace(n_classes=10)
            graph.sample_architecture(op_indces=op)
            graph.parse()
            graph.get_hash()
            dataset_apis={}
            dataset_apis["NASBench201-cifar10"] = get_dataset_api(search_space='nasbench201', dataset='cifar10')
            train_acc_parent = graph.query(metric=Metric.TRAIN_ACCURACY, dataset='cifar10', dataset_api=dataset_apis["NASBench201-cifar10"])
            val_acc_parent = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=dataset_apis["NASBench201-cifar10"])
            
            train_accuries.append(train_acc_parent)
            val_accuries.append(val_acc_parent)


            for zc_proxy in proxy_config:
                zc_predictor = ZeroCost(method_type=zc_proxy)
                score = zc_predictor.query(graph=graph, dataloader=train_loader)
                scores[proxy_name].append(score)

        np.save(scores)
        np.save(train_accuries)
        np.save(val_accuries)

        return
    
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
