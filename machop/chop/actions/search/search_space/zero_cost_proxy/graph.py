# This is the search space for Zero Cost Proxies

from ..base import SearchSpaceBase
import logging

from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api, get_zc_benchmark_api
from naslib.utils import get_train_val_loaders, get_project_root
from fvcore.common.config import CfgNode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBRegressor
import numpy as np
from .utils import sample_arch_dataset, evaluate_predictions, eval_zcp

logger = logging.getLogger(__name__)

DEFAULT_ZERO_COST_PROXY_CONFIG = {
    "config": {
        "benchmark": "nasbench201",
        "dataset": "cifar10",
        "num_archs_train": 2000,
        "num_archs_test": 2000,
        "calculate_proxy": False,
        "loss_fn": "mae",
        "optimizer": "adam",
        "zc_proxies": [
            "epe_nas",
            "fisher",
            "grad_norm",
            "grasp",
            "jacov",
            "l2_norm",
            "nwot",
            "plain",
            "snip",
            "synflow",
            "zen",
            "flops",
            "params",
        ],
    }
}


class ZeroCostProxy(SearchSpaceBase):
    """
    Zero Cost Proxy search space.
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_ZERO_COST_PROXY_CONFIG
        self.zcp_results = []

    def calculate_zc(self, xtrain, xtest, ytrain, ytest, zc_api):
        # Create configs required for get_train_val_loaders
        config_dict = {
            "dataset": self.config["zc"][
                "dataset"
            ],  # Dataset to loader: can be cifar10, cifar100, ImageNet16-120
            "data": str(get_project_root())
            + "/data",  # path to naslib/data where the data is saved
            "search": {
                "seed": self.config["zc"][
                    "seed"
                ],  # Seed to use in the train, validation and test dataloaders
                "train_portion": 0.7,  # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
                "batch_size": 32,  # batch size of the dataloaders
                "cutout": False,
            },
        }
        config = CfgNode(config_dict)

        for zcp_name in self.config["zc"]["zc_proxies"]:
            if self.config["zc"]["calculate_proxy"]:
                if self.config["zc"]["dataset"] == "ImageNet16-120":
                    raise ValueError("Please set 'calculate_proxy' to false")
                # Get the dataloaders
                train_loader, _, _, _, _ = get_train_val_loaders(config)
                # train and query expect different ZCP formats
                zcp_train = [
                    {"zero_cost_scores": eval_zcp(t_arch, zcp_name, train_loader)}
                    for t_arch in xtrain
                ]
                zcp_test = [
                    {"zero_cost_scores": eval_zcp(t_arch, zcp_name, train_loader)}
                    for t_arch in xtest
                ]
                zcp_pred_test = [s["zero_cost_scores"][zcp_name] for s in zcp_test]
                zcp_pred_train = [s["zero_cost_scores"][zcp_name] for s in zcp_train]
            else:
                zcp_train = [
                    {"zero_cost_scores": zc_api[str(t_arch)][zcp_name]["score"]}
                    for t_arch in xtrain
                ]
                zcp_test = [
                    {"zero_cost_scores": zc_api[str(t_arch)][zcp_name]["score"]}
                    for t_arch in xtest
                ]

                zcp_pred_test = [s["zero_cost_scores"] for s in zcp_test]
                zcp_pred_train = [s["zero_cost_scores"] for s in zcp_train]

            train_metrics = evaluate_predictions(ytrain, zcp_pred_train)
            test_metrics = evaluate_predictions(ytest, zcp_pred_test)

            results = []
            for i, t_arch in enumerate(xtest):
                results.append(
                    {
                        "test_hash": f"{t_arch}",
                        "test_accuracy": ytest[i],
                        "zc_metric": zcp_pred_test[i],
                    }
                )

            self.zcp_results.append(
                {
                    zcp_name: {
                        "test_spearman": test_metrics["spearmanr"],
                        "train_spearman": train_metrics["spearmanr"],
                        "test_kendaltau": test_metrics["kendalltau"],
                        "results": results,
                    }
                }
            )

    def build_search_space(self):
        """
        Build the search space zero cost proxies
        """

        seed = self.config["zc"]["seed"]
        pred_dataset = self.config["zc"]["dataset"]

        # get data from api
        zc_api = get_zc_benchmark_api(self.config["zc"]["benchmark"], pred_dataset)
        pred_api = get_dataset_api(
            search_space=self.config["zc"]["benchmark"],
            dataset=self.config["zc"]["dataset"],
        )
        train_size = self.config["zc"]["num_archs_train"]
        test_size = self.config["zc"]["num_archs_test"]

        # get train and test architectures
        train_sample, train_hashes = sample_arch_dataset(
            NasBench201SearchSpace(),
            pred_dataset,
            pred_api,
            data_size=train_size,
            shuffle=True,
            seed=seed,
        )
        test_sample, _ = sample_arch_dataset(
            NasBench201SearchSpace(),
            pred_dataset,
            pred_api,
            arch_hashes=train_hashes,
            data_size=test_size,
            shuffle=True,
            seed=seed + 1,
        )

        # get train and test samles
        xtrain, ytrain, _ = train_sample
        xtest, ytest, _ = test_sample

        self.calculate_zc(xtrain, xtest, ytrain, ytest, zc_api)
