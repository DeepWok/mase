# This is the search space for pruning a network. It is used in the iterative pruning strategy.
from math import ceil

import torch

from ..base import SearchSpaceBase
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_pruning_metadata_analysis_pass,
    prune_transform_pass,
    add_software_metadata_analysis_pass
)
from .....passes.graph.utils import get_mase_op
from ..utils import flatten_dict, unflatten_dict
from ...strategies.runners.software import get_sw_runner
from .....dataset import MaseDataModule, get_dataset_info
from .....models import get_model


DEFAULT_PRUNE_CONFIG = {
    "config": {
        "prune_config": {
            "num_iterations": 1,
            "scope": "global",
            "granularity": "elementwise",
            "method": "l1-norm",
            "sparsity": 0.5,
        },
        "train_config": {
            "max_epochs": 5,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "name": "accuracy",
            "data_loader": "train_dataloader",
            "num_samples": 100000,
            "num_warmup_steps": 0,
            "lr_scheduler": "linear",
        }
    }
}

SET_TRAIN_PARAMS = {
    "name": ["accuracy"],
    "data_loader": ["train_dataloader"],
    "num_samples": [100000],
    "num_warmup_steps": [0],
    "lr_scheduler": ["linear"],
}

class IterativePruningSpace(SearchSpaceBase):
    """
    Search space for pruning a network
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_PRUNE_CONFIG

        self.config = self.config.get("config")

        self.model_config = self.config.get("model_config")

        data_module = MaseDataModule(
            name=self.model_config["dataset"],
            batch_size=self.model_config["batch_size"],
            model_name=self.model_config["model"],
            num_workers=0,
        )
        data_module.prepare_data()
        data_module.setup()

        self.data_module = data_module

        self.dataset_info = get_dataset_info(self.model_config["dataset"])

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        # set train/eval mode before creating mase graph
        self.model = get_model(
            self.model_config['model'],
            self.model_config['task'],
            self.dataset_info,
            pretrained=False,
        )
        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        # Split the passed config into the pruning and training configs
        pruning_config = sampled_config['iterative_prune']
        training_config = sampled_config['train']

        overall_sparsity = pruning_config["sparsity"]
        num_iterations = pruning_config["num_iterations"]

        dummy_in = {"x": next(iter(self.data_module.train_dataloader()))[0]}

        max_epochs = training_config["max_epochs"]
        epochs_per_iteration = ceil(max_epochs / num_iterations)
        training_config["max_epochs"] = epochs_per_iteration

        # Fetch the runner for training the network
        train_runner = get_sw_runner(
            "basic_train",
            self.model_info,
            self.model_config['task'],
            self.dataset_info,
            self.accelerator,
            training_config,
        )
        
        # Initialize the mase graph
        mg = MaseGraph(self.model)
        mg, _ = init_metadata_analysis_pass(mg, dummy_in)
        mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in, "force_device_meta": False})
        mg, _ = add_software_metadata_analysis_pass(mg, None)

        # Set the base pruning args
        prune_args = {
            "weight": {
                "scope": pruning_config["scope"],
                "granularity": pruning_config["granularity"],
                "method": pruning_config["method"],
            },
            "activation": {
                "scope": pruning_config["scope"],
                "granularity": pruning_config["granularity"],
                "method": pruning_config["method"],
            },
        }
        
        # Save the original weights and biases. These will be used to reset the model after each iteration.
        original_w_b = {}
        for node in mg.fx_graph.nodes:
            if get_mase_op(node) in ["linear", "conv2d", "conv1d"]:
                original_w_b[node.name] = {
                    "weight": mg.modules[node.target].weight,
                    "bias": mg.modules[node.target].bias,
                    "meta_weight": node.meta["mase"].parameters["common"]["args"]["weight"]["value"],
                    "meta_bias": node.meta["mase"].parameters["common"]["args"]["bias"]["value"],
                }

        train_metrics = []

        # Prune the model iteratively
        for i in range(num_iterations):
            results = train_runner(self.data_module, self.model, None)
            train_metrics.append(results)

            # Calculate the sparsity for the current iteration
            iteration_sparsity = 1 - (1-overall_sparsity)**((i+1)/num_iterations)

            # Update the sparsity in the prune args
            prune_args["weight"]["sparsity"] = iteration_sparsity
            prune_args["activation"]["sparsity"] = iteration_sparsity

            # Prune the model
            mg, _ = prune_transform_pass(mg, prune_args)

            # Copy the original weights and biases back to the model
            for node in mg.fx_graph.nodes:
                if get_mase_op(node) in ["linear", "conv2d", "conv1d"]:
                    with torch.no_grad():
                        mg.modules[node.target].weight.copy_(original_w_b[node.name]['weight'])
                        mg.modules[node.target].bias.copy_(original_w_b[node.name]['bias'])
                        
            # Run the analysis passes
            mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in, "force_device_meta": False})
            mg, _ = add_software_metadata_analysis_pass(mg, None)
            mg, _ = add_pruning_metadata_analysis_pass(mg, {"dummy_in": dummy_in, "add_value": True})

        self.model.to(self.accelerator)

        return mg

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for the prune search
        """

        # Build the search space
        choices = self.config.get("search_config", DEFAULT_PRUNE_CONFIG)
        
        for k, v in SET_TRAIN_PARAMS.items():
            choices['train'][k] = v

        # flatten the choices and choice_lengths
        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }

    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        """
        Convert sampled flattened indexes to a nested config which will be passed to `rebuild_model`.

        ---
        For example:
        ```python
        >>> indexes = {
            "conv1/config/name": 0,
            "conv1/config/bias_frac_width": 1,
            "conv1/config/bias_width": 3,
            ...
        }
        >>> choices_flattened = {
            "conv1/config/name": ["integer", ],
            "conv1/config/bias_frac_width": [5, 6, 7, 8],
            "conv1/config/bias_width": [3, 4, 5, 6, 7, 8],
            ...
        }
        >>> flattened_indexes_to_config(indexes)
        {
            "conv1": {
                "config": {
                    "name": "integer",
                    "bias_frac_width": 6,
                    "bias_width": 6,
                    ...
                }
            }
        }
        """
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        return config
