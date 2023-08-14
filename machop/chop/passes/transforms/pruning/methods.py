"""
Pruning methods (incl. add-ons)

----------------------------------------------------------------------------------------
Internal Backlog:
1. The benefit of the adaptive strategy is that you directly control the minimum
   sparsity in activations. A neat extension would be to save the thresholds that it
   generates via a forward pass (invoking the hook on a representative batch) and use
   it for a fixed-layerwise strategy. It doesn't guarantee the minimum sparsity, but
   it's close enough.
"""

from collections import OrderedDict
from functools import partial
from tabulate import tabulate
from abc import ABC, abstractmethod
import toml

import torch
import torch.nn as nn

from chop.tools.logger import getLogger

# Housekeeping -------------------------------------------------------------------------
logger = getLogger(__name__)
logger.propagate = False  # Avoids duplicate logging messages


# Pruning routines ---------------------------------------------------------------------
# The base pruner class that all pruners should inherit from.
# NOTE: When creating a new pruner, please make sure to follow this short checklist:
# 1. Add the activation handler as a class attribute.
# 2. Always check if the handler is None because it's not guaranteed to be passed in.
# 3. Make sure to store the handle for this hook in the handler's handles dictionary.
# --------------------------------------------------------------------------------------
# If you're not supporting activation pruning, then make it explicit to the user. The
# graph iterator will always pass it in if the user requests it. If the handles aren't
# stored, then unwrapping (i.e. remvoing the hooks) wouldn't work via the handler's
# unwrap method and would require manual intervention to clear the hooks.
class BasePruner(ABC):
    def __init__(self, sparsity, criterion, handler: "ActivationPruneHandler" = None):
        self.sparsity = sparsity
        self.criterion = criterion
        self.handler = handler
        self.summary = []

    # Abstract methods -----------------------------------------------------------------
    # Python doesn't check the method signature, but please make sure to follow the
    # signatures below when overriding these methods; the callsite (graph iterator)
    # expects the methods to have them.
    @abstractmethod
    def wrap(self, module: nn.Module, name: str):
        # Register the pre-forward hook for activation pruning here. :)
        pass

    @abstractmethod
    def apply(self, module: nn.Module, name: str, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def unwrap(self, module: nn.Module, name: str):
        pass

    @abstractmethod
    def show_summary(self):
        pass

    @abstractmethod
    def save_summary(self, save_dir):
        pass


# NOTE: For now, we restrict pruning to the local scope (i.e. layer-wise) and only
# support a one-shot schedule. The level pruner is arguably the simplest pruner.
class LevelPruner(BasePruner):
    # Fields to show in the summary
    FIELDS = (
        "Layer",
        "Shape",
        "Sparsity",
        "Total",
        "NNZ",
    )

    def wrap(self, *_):
        # NOTE: This is where you'd register the initial weight mask but our use case
        # doesn't require addtional parameters or buffers.
        pass

    def apply(self, module: nn.Module, name: str, **kwargs):
        # NOTE: Since we're not fine-tuning the model, we don't need to store the
        # original weights and weight mask as registered parameters or buffers. :)
        weight = module.weight.data.clone()
        mask = (
            self.criterion(weight, self.sparsity, **kwargs)
            if self.sparsity > 0.0
            else torch.ones_like(weight, dtype=torch.bool)
        )
        module.weight.data.mul_(mask)
        self._update_summary(module, name)

    # NOTE: Currently, this method is unused.
    # Its purpose is to remove additional parameters or buffers that were added to the
    # model for weight pruning. However, we don't currently do this. Since the model
    # may be saved with masks, this method is static as we'd lose the pruner instance.
    # The idea is that the pruner used to prune a module should handle the unwrapping.
    # ----------------------------------------------------------------------------------
    # Also, this method shoudn't concern cleaning up activation pruning hooks as the
    # handler object is saved with the model; its unwrap method should be used instead,
    # as in the unwrap pass.
    @staticmethod
    def unwrap(*_):
        pass

    # Show the layerwise summary of the changes in sparsity resulting from pruning
    def show_summary(self):
        colalign = ("left", "center")
        logger.info(
            f"\n{tabulate(self.summary, self.FIELDS, 'pretty', colalign=colalign)}"
        )

    # Save the pruning summary to a CSV file
    def save_summary(self, save_dir):
        save_path = save_dir / "weight_summary.csv"
        with open(save_path, "w") as f:
            f.write(",".join(self.FIELDS) + "\n")
            for row in self.summary:
                f.write(",".join(map(str, row)) + "\n")
        logger.info(f"Saved weight pruning summary to {save_path}")

    # "Private" helper methods ---------------------------------------------------------
    def _update_summary(self, module: nn.Module, name: str):
        tot_elements = module.weight.numel()
        nnz_elements = torch.count_nonzero(module.weight)
        sparsity = f"{1 - nnz_elements / tot_elements:.4f}"
        self.summary.append(
            [
                name,
                f"({', '.join(map(str, list(module.weight.shape)))})",
                sparsity,
                f"{tot_elements:,}",
                f"{nnz_elements:,}",
            ]
        )


# Activation pruner routine ------------------------------------------------------------
# Strategies to prune activations
ACTIVATION_PRUNE_STRATEGIES = [
    # Do nothing; this is useful for collecting statistics on how weights induce
    # sparsity in activations; setting the weight sparsity target to 0 allows one to
    # also observe the sparsity in activation for an unpruned model. :)
    "observe",
    # Generate the threshold on the fly to meet the desired sparsity target; this is
    # highly similar to how level pruning works.
    "adaptive",
    # A fixed manual threshold that is applied across all activations; this is a naive
    # approach as two problems may arise:
    # 1. The threshold is too small than the range of values in the activation that
    #    little-to-no values are pruned.
    # 2. The threshold is larger than the entire range of values in the activation that
    #    nearly all values in the activation is pruned, corrupting model quality
    # NOTE: This is where a calibration pass is essential and motivates a layerwise
    # threshold that may be fine-tuned. :)
    "fixed-global",
    "fixed-layerwise"  # A dictionary where the key is the target node's name :)
    # More here... (e.g. adaptive-layerwise)
]


# The activation pruner is like an add-on to the weight pruner :)
# NOTE: As the handler is saved with the model, it's crucial that the class attributes
# don't change or you'll break compatibility; class methods are fine though. Notably,
# when an class instance is pickled, its class code isn't pickled; only the instance
# attributes are. This is by design. Please see the following for more information:
# https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
# --------------------------------------------------------------------------------------
# When loading the model, make sure to only use the graph module (mz) and not the state
# dictionary (pt) as the latter only saves parameters and buffers, so all hooks are lost
# along with the handler object.
class ActivationPruneHandler:
    # Fields to show in the summary
    FIELDS = (
        "Layer",
        "Shape",
        "Mean Sparsity (Old -> New)",
        "Total",
        "Mean NNZ (Old -> New)",
        "Threshold",
        "Min / Median / Max",
    )

    def __init__(self, strategy, target):
        self.strategy = strategy
        self.target = target
        # A dictionary of tuples where the key is the name of the node and the value is
        # a tuple of two lists: the average and variance of the activation sparsity
        # across all channels. That is, (assuming three channels)
        # { ..., "feature.0.conv1": ([0.1, 0.2, 0.3], [0.01, 0.02, 0.03]), ... }
        self.report = OrderedDict()
        # Misc. information on layerwise activation sparsity
        self.summary = []
        # Node-specific thresholds that are logged when the hook is called.
        # NOTE: Everytime a forward pass is invoked, the dictionary is rewritten. It's
        # only for the adaptive strategy that the values stored will actually change.
        self.thresholds = {}
        # Handles for pre-forward hooks; added by the pruner when the hook is registered
        self.handles = OrderedDict()
        # Toggle logging metadata
        self.log_report = True
        self.log_summary = True
        self.log_thresholds = True

    # When calling, use partial to bind the name of the module and the activation prune
    # handler object to the hook function. Also, when saving, make sure to add the
    # handler object as an attribute of the model so that it doesn't become a dangling
    # reference. :)
    @staticmethod
    def hook(name, handler, _, args):
        # NOTE: We only expect one input to the module! Also, the threshold is
        # None if the strategy is "observe".
        input = args[0]
        threshold = 0
        if handler.strategy == ACTIVATION_PRUNE_STRATEGIES[1]:  # adaptive
            threshold = torch.quantile(input.abs().flatten(), handler.target)
        if handler.strategy == ACTIVATION_PRUNE_STRATEGIES[2]:  # fixed-global
            threshold = handler.target
        elif handler.strategy == ACTIVATION_PRUNE_STRATEGIES[3]:  # fixed-layerwise
            threshold = handler.target[name]

        # Generate the mask
        # NOTE: For the observe strategy, this mask captures the sparsity in the input.
        mask = (input.abs() > threshold).to(input.dtype)

        # Compute the channel-wise sparsity of the mask and take the mean and variance
        # along the batch dimension.
        tot_elements = mask.size(2) * mask.size(3)
        nnz_elements = torch.sum(mask, dim=(2, 3))
        channel_sparsity = 1 - nnz_elements / tot_elements
        sparsity_avg = torch.mean(channel_sparsity, dim=0).tolist()
        sparsity_var = torch.var(channel_sparsity, dim=0).tolist()

        if handler.log_thresholds:
            handler.thresholds[name] = threshold
        if handler.log_report:
            handler.report[name] = {"avg": sparsity_avg, "var": sparsity_var}
        if handler.log_summary:
            handler._update_summary(name, threshold, input, mask)

        # Apply the mask to the input (magic happens here)
        return input * mask

    def wrap(self, module: nn.Module, name: str):
        # We only care about enforcing or observing activation sparsity in conv2d layers
        if not isinstance(module, nn.Conv2d):
            return
        # Skip registering the hook if the node isn't a key
        if self.strategy == ACTIVATION_PRUNE_STRATEGIES[3] and name not in self.target:
            return
        self.handles[name] = module.register_forward_pre_hook(
            partial(self.hook, name, self)
        )

    def unwrap(self):
        for handle in self.handles.values():
            handle.remove()

    def show_summary(self):
        colalign = ("left", "center")
        logger.info(
            f"\n{tabulate(self.summary, self.FIELDS, 'pretty', colalign=colalign)}"
        )

    # Save the activation pruning summary to a CSV file
    def save_summary(self, save_dir):
        save_path = save_dir / "activation_summary.csv"
        with open(save_path, "w") as f:
            f.write(",".join(self.FIELDS) + "\n")
            for row in self.summary:
                f.write(",".join(map(str, row)) + "\n")
        logger.info(f"Saved activation pruning summary to {save_path}")

    # Save the channel-wise activation sparsity report to a TOML file
    def save_report(self, save_dir):
        save_path = save_dir / "activation_report.toml"
        with open(save_path, "w") as f:
            toml.dump(self.report, f)
        logger.info(f"Saved activation pruning report to {save_path}")

    # Cleanup --------------------------------------------------------------------------
    # NOTE: This is useful for a fresh start. Please use it before running inference.
    def clear_report(self):
        self.report = OrderedDict()

    def clear_summary(self):
        self.summary = []

    def clear_thresholds(self):
        self.thresholds = {}

    # "Private" helper methods ---------------------------------------------------------
    def _update_summary(self, name, threshold, input, mask):
        # Compute the sparsity of the feature maps in the input and take the average
        # along both the channel and batch dimensions.
        tot_elements = input.size(2) * input.size(3)
        nnz_elements_old = torch.sum(input.abs() > 0, dim=(2, 3), dtype=torch.float)
        nnz_elements_new = torch.sum(mask, dim=(2, 3), dtype=torch.float)
        sparsity_old = 1 - nnz_elements_old / tot_elements
        sparsity_new = 1 - nnz_elements_new / tot_elements
        self.summary.append(
            [
                name,
                f"({', '.join(map(str, list(input.shape)))})",
                f"{sparsity_old.mean():.4f} -> {sparsity_new.mean():.4f}",
                f"{tot_elements:,}",
                f"{nnz_elements_old.mean():.2f} -> {nnz_elements_new.mean():.2f}",
                f"{threshold:.4f}" if threshold is not None else "N/A",
                f"{input.min():.4f} / {input.median():.4f} / {input.max():.4f}",
            ]
        )
