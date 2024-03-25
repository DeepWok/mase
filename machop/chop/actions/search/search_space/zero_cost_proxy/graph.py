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

from .models import ZeroCostLinearModel, ZeroCostNonLinearModel
from .utils import sample_arch_dataset, evaluate_predictions, eval_zcp

logger = logging.getLogger(__name__)


DEFAULT_ZERO_COST_PROXY_CONFIG = {
    "config": {
        "benchmark": "nasbench201",
        "dataset": "cifar10",
        "num_archs_train": 2000,
        "num_archs_test": 2000,
        "calculate_proxy": False,
        "ensemble_model": "nonlinear",
        "loss_fn": "mae",
        "optimizer": "adam",
        "batch_size": 4,
        "learning_rate": 0.02,
        "epochs": 30,
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
        self.custom_ensemble_metrics = {}
        self.xgboost_metrics = {}

    def train_zc_ensemble_model(
        self, inputs_train, targets_train, inputs_test, targets_test
    ):
        logger.info("Training Custom Neural Network")

        class CustomDataset(Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]

        # Convert lists to PyTorch tensors
        inputs_train_tensor = torch.tensor(inputs_train, dtype=torch.float32)
        targets_train_tensor = torch.tensor(targets_train, dtype=torch.float32).view(
            -1, 1
        )
        inputs_test_tensor = torch.tensor(inputs_test, dtype=torch.float32)
        targets_test_tensor = torch.tensor(targets_test, dtype=torch.float32).view(
            -1, 1
        )

        # Create dataset instances
        train_dataset = CustomDataset(inputs_train_tensor, targets_train_tensor)
        test_dataset = CustomDataset(inputs_test_tensor, targets_test_tensor)

        batch_size = self.config["zc"]["batch_size"]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        ensemble_model = self.config["zc"]["ensemble_model"]
        input_size = len(inputs_train[0])
        try:
            if ensemble_model == "linear":
                model = ZeroCostLinearModel(input_size)
            elif ensemble_model == "nonlinear":
                model = ZeroCostNonLinearModel(input_size)
        except:
            raise ValueError(
                f"Unknown model type: {ensemble_model}. Has to be one of linear or nonlinear"
            )

        loss = self.config["zc"]["loss_fn"]
        try:
            if loss == "mse":
                criterion = nn.MSELoss()
            elif loss == "mae":
                criterion = nn.L1Loss()
            elif loss == "huber":
                criterion = nn.SmoothL1Loss()
        except:
            raise ValueError(
                f"Unknown criterion type: {loss}. Has to be one of mse, mae or huber"
            )

        opt = self.config["zc"]["optimizer"]
        lr = self.config["zc"]["learning_rate"]

        try:
            if opt == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif opt == "adamW":
                optimizer = optim.AdamW(model.parameters(), lr=lr)
            elif opt == "rmsProp":
                optimizer = optim.RMSprop(model.parameters(), lr=lr)
        except:
            raise ValueError(
                f"Unknown optimizer type: {opt}. Has to be one of adam, adamW or rmsProp"
            )

        optimizer = optim.Adam(model.parameters(), lr=lr)

        epochs = self.config["zc"]["epochs"]
        best_test_loss = float("inf")  # Initialize best test loss to a high value
        best_model_state = None  # To store the best model state

        for epoch in range(epochs):
            model.train()
            running_loss_train = 0.0
            for inputs_batch, targets_batch in train_loader:
                optimizer.zero_grad()
                outputs_train = model(inputs_batch)
                loss_train = criterion(outputs_train, targets_batch)
                loss_train.backward()
                optimizer.step()
                running_loss_train += loss_train.item() * inputs_batch.size(0)

            epoch_loss_train = running_loss_train / len(train_loader.dataset)

            # Evaluation mode (no gradients)
            model.eval()
            running_loss_test = 0.0
            with torch.no_grad():
                for inputs_batch, targets_batch in test_loader:
                    outputs_test = model(inputs_batch)
                    loss_test = criterion(outputs_test, targets_batch)
                    running_loss_test += loss_test.item() * inputs_batch.size(0)

            epoch_loss_test = running_loss_test / len(test_loader.dataset)

            # Check if this is the best model based on test loss and update accordingly
            if epoch_loss_test < best_test_loss:
                best_test_loss = epoch_loss_test
                best_model_state = (
                    model.state_dict().copy()
                )  # Save a copy of the best model state

        # After training, load the best model state back into your model
        model.load_state_dict(best_model_state)

        return model

    def test_zc_ensemble_model(self, model, inputs_test, ytest):
        model.eval()
        predicted_accuracies = []
        with torch.no_grad():  # No need to track gradients
            for i in range(
                len(inputs_test)
            ):  # Assuming you want to use all test inputs
                predicted_accuracy = model(
                    torch.Tensor(inputs_test[i])
                )  # Add batch dimension
                predicted_accuracies.append(predicted_accuracy.item())

        self.custom_ensemble_metrics = evaluate_predictions(ytest, predicted_accuracies)

    def get_model_inputs(self, dataset, archs, zc_proxies):
        inputs = []
        for arch in archs:
            inputs.append(
                [
                    dataset[str(arch)].get(metric_name, 0)["score"]
                    for metric_name in zc_proxies
                ]
            )
        return inputs

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
                "cutout": False
            }
        }
        config = CfgNode(config_dict)


        for zcp_name in self.config["zc"]["zc_proxies"]:
            if self.config["zc"]["calculate_proxy"]:
                if self.config["zc"]["dataset"] == 'ImageNet16-120':
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

        # prepare the inputs and targets based on the modified combined_data_list
        inputs_train = self.get_model_inputs(
            zc_api, xtrain, self.config["zc"]["zc_proxies"]
        )
        inputs_test = self.get_model_inputs(
            zc_api, xtest, self.config["zc"]["zc_proxies"]
        )

        # neural network model
        model = self.train_zc_ensemble_model(inputs_train, ytrain, inputs_test, ytest)
        self.test_zc_ensemble_model(model, inputs_test, ytest)

        # XGBoost model
        X_train = np.array(inputs_train)
        X_test = np.array(inputs_test)
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train, ytrain)
        xgb_preds = xgb_model.predict(X_test)
        self.xgboost_metrics = evaluate_predictions(ytest, xgb_preds)

        # calculate zc proxies
        self.calculate_zc(xtrain, xtest, ytrain, ytest, zc_api)
