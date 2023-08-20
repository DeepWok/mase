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
from pathlib import Path
import toml
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from chop.tools.logger import getLogger
from chop.passes.transforms.pruning.utilities import StatisticsCollector

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
                # Wrap everything in quotes to avoid issues with commas in the data
                f.write(",".join(map(str, [f'"{x}"' for x in row])) + "\n")
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

    def __init__(self, target, strategy, save_dir):
        self.target = target
        self.strategy = strategy
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Internal Attributes ----------------------------------------------------------
        # Verify that the custom conv implementation is indeed correct (within 1e-5)
        self.verify = True
        # The number of samples to randomly sample for the statistics collector. Set it
        # to None to collect all samples.
        self.samples = 1000
        # Misc. information for the console summary (activation pruning)
        self.summary = []
        # Handles for hooks. This is actually nested dictionary handles for each node.
        # We have one pre-forward ("pre") and forward ("fwd") hook for each node.
        self.handles = OrderedDict()
        # A dict. of StatisticCollector objects, with the key being the node's name.
        # The object logs the statistics of window-wise sparsities of post-product
        # feature maps (before accumulation) across all input channels.
        self.statistics = OrderedDict()
        # Node-specific thresholds that are logged when the prune hook is called.
        # NOTE: Everytime a forward pass is invoked, the dictionary is rewritten. It's
        # only for the adaptive strategy that the values stored will actually change.
        self.thresholds = {}

        # Logging ----------------------------------------------------------------------
        # NOTE: log_input, log_values and verify are linked with log_statistics. See
        # the collect hook and _conv_forward function for more details.
        self.log_input = True
        self.log_values = True
        self.log_summary = True
        self.log_statistics = True
        self.log_thresholds = True

    # Hooks ----------------------------------------------------------------------------
    # When calling, use partial to bind the name of the module and the activation prune
    # handler object to the hook functions. When saving, add the handler object as an
    # attribute of the model so that it doesn't become a dangling reference. :)
    @staticmethod
    def prune(name, handler, _, args):
        # NOTE: We only expect one input to the module!
        input = args[0]
        if handler.strategy == ACTIVATION_PRUNE_STRATEGIES[0]:  # observe
            return input

        threshold = 0.0
        if handler.strategy == ACTIVATION_PRUNE_STRATEGIES[1]:  # adaptive
            threshold = torch.quantile(input.abs().flatten(), handler.target)
        if handler.strategy == ACTIVATION_PRUNE_STRATEGIES[2]:  # fixed-global
            threshold = handler.target
        elif handler.strategy == ACTIVATION_PRUNE_STRATEGIES[3]:  # fixed-layerwise
            threshold = handler.target[name]

        # Generate the mask
        mask = (input.abs() > threshold).to(input.dtype)

        # Log the threshold if necessary; for the observe strategy, we don't do this.
        # The summary, on the other hand, only captures sparsity enforced by pruning.
        if handler.log_thresholds:
            handler.thresholds[name] = threshold
        if handler.log_summary:
            handler._update_summary(name, threshold, input, mask)

        # Apply the mask! This is where the magic happens :)
        return input * mask

    @staticmethod
    def collect(name, handler: "ActivationPruneHandler", module, args, output):
        # Deepcopy the module, detach the input and feed it through the custom forward
        if handler.log_statistics:
            ref = output
            out = handler._conv_forward(copy.deepcopy(module), name, args[0].detach())
            if handler.verify:
                e = torch.abs(ref - out)
                avg = f"{torch.mean(e):.4e}"
                max = f"{torch.max(e):.4e}"
                logger.debug(f"(Avg Error: {avg}, Max Error: {max}) for {name}")
                assert torch.allclose(ref, out, atol=1e-4), "Forward pass mismatch!"

    # Main Methods ---------------------------------------------------------------------
    def wrap(self, module: nn.Module, name: str):
        # We only care about enforcing or observing activation sparsity in nn.Conv2d
        # layers and we don't want to register the hooks if the node isn't a key
        if not isinstance(module, nn.Conv2d):
            return
        if self.strategy == ACTIVATION_PRUNE_STRATEGIES[3] and name not in self.target:
            return

        # Register the hooks!
        self.handles[name] = {
            "pre": module.register_forward_pre_hook(partial(self.prune, name, self)),
            "fwd": module.register_forward_hook(partial(self.collect, name, self)),
        }

    def unwrap(self, keep_pre=False):
        for handles in self.handles.values():
            if not keep_pre:
                handles["pre"].remove()
            handles["fwd"].remove()

    # Logging --------------------------------------------------------------------------
    def show_summary(self):
        if self.summary == []:
            logger.info("No records! Observe strategy doesn't enforce sparsity.")
        else:
            colalign = ("left", "center")
            table = tabulate(self.summary, self.FIELDS, "pretty", colalign=colalign)
            logger.info(f"\n{table}")

    # Save the activation pruning summary to a CSV file
    def save_summary(self, save_dir):
        save_path = save_dir / "activation_summary.csv"
        with open(save_path, "w") as f:
            f.write(",".join(self.FIELDS) + "\n")
            for row in self.summary:
                # Wrap everything in quotes to avoid issues with commas in the data
                f.write(",".join(map(str, [f'"{x}"' for x in row])) + "\n")
        logger.info(f"Saved activation pruning summary to {save_path}")

    # Save the logged statistics to a few files
    def save_statistics(self, save_dir):
        save_path = save_dir / "activation_report.toml"
        stats_dir = save_dir / "statistics"  # This is where the pickled objects go
        stats_dir.mkdir(parents=True, exist_ok=True)

        report = {}
        for name, statistics in self.statistics.items():
            report[name] = {
                "avg": statistics.avg.cpu().tolist(),
                "var": statistics.var.cpu().tolist(),
            }
            torch.save(
                {
                    "avg": statistics.avg.cpu(),
                    "var": statistics.var.cpu(),
                    "cov": statistics.cov.cpu(),
                    "cor": statistics.cor.cpu(),
                },
                stats_dir / f"{name}.pt",
            )

        with open(save_path, "w") as f:
            toml.dump(report, f)

        logger.info(f"Saved activation pruning report to {save_path}")

    # Cleanup --------------------------------------------------------------------------
    # NOTE: This is useful for a fresh start. Please use it before running inference.
    def clear_statistics(self):
        self.statistics = OrderedDict()

    def clear_summary(self):
        self.summary = []

    def clear_thresholds(self):
        self.thresholds = {}

    # "Private" helper methods ---------------------------------------------------------
    # This is the Python implementation of a convolutional layer's forward pass that the
    # collect hook calls to, well, collect statistics of window-wise patches along
    # each channel of the post-product feature maps. This will be slow, as expected.
    def _conv_forward(self, module: nn.Module, name: str, x: torch.Tensor):
        assert x.size()[2] == x.size()[3], "Only square feature maps are supported!"

        # The saved input is used for validating the forward pass in hardware
        if self.log_input:
            input_dir = self.save_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            torch.save(x.detach(), input_dir / f"{name}.pt")

        # Parameters of a convolutional layer
        # NOTE: ic in the weights is the number of input channels per group
        oc, ic, kh, kw = module.weight.shape
        gs = module.groups  # No. of groups
        ic *= gs
        bs = x.shape[0]

        # Each patch from an input channel is processed by oc // gs kernels, so we
        # expand (i.e repeat) the patches accordingly.
        # NOTE: The shape after the permute is (bs, hs, ws, oc, ic, kh, kw), where the
        # oc is the number of output channels per group. However, ic is the total
        # number of input channels; this will be dealt with later.
        p, s, d = module.padding, module.stride, module.dilation
        hs = ws = (x.size(2) + 2 * p[0] - d[0] * (kh - 1) - 1) // s[0] + 1
        patches = (
            F.unfold(x, (kh, kw), d, p, s)
            .transpose(1, 2)
            .reshape(bs, hs, ws, ic, kh, kw)
            .expand(oc // gs, *(bs, hs, ws, ic, kh, kw))
            .permute(1, 2, 3, 0, 4, 5, 6)
            .contiguous()
        )

        # Create the output tensor on the same device as the module
        out = torch.zeros((bs, hs, ws, oc), device=module.weight.device)

        # We only roll part of the patches tensor to reduce memory usage, and rf (i.e
        # the roll factor) is the number of times we roll the tensor.
        rf = 2 if hs % 2 == 0 else self._get_smallest_factor(hs)
        assert hs % rf == 0, "Roll factor not compatible!"
        for hi, wi in np.ndindex(rf, rf):
            # s is the shfit size; s * s would be the number of patches we process in
            # one loop iteration. It's easy to think of this as a sliding window.
            s = hs // rf
            patch = (
                patches[:, hi * s : (hi + 1) * s, wi * s : (wi + 1) * s]
                # This is where we deal with the ic issue mentioned above; we then
                # tranpose the tensor to push the gs dim. before the oc dim.
                .reshape(bs, s, s, oc // gs, gs, ic // gs, kh, kw)
                .transpose(3, 4)
                .mul(module.weight.reshape(gs, oc // gs, ic // gs, kh, kw))
            )

            # Log statistics about the post-product patches
            # NOTE: Since there are a large number of patches, computing the statistics
            # across all of them will be expensive. To reduce the cost, we can compute
            # the statistics for a random subset.
            if self.samples is not None and self.samples < bs * s * s * oc:
                size = bs * s * s * oc
                idxs = torch.randint(0, size, (self.samples,), device=patch.device)
                temp = patch.detach().view(-1, ic // gs, kh * kw)[idxs]
            else:
                temp = patch.detach().view(-1, ic // gs, kh * kw)

            sparsity = 1 - temp.count_nonzero(dim=2) / (kh * kw)
            if name not in self.statistics:
                device = module.weight.device
                sc = StatisticsCollector(name, ic // gs, device, self.log_values)
                self.statistics[name] = sc
            self.statistics[name].update(sparsity, f"{hi}_{wi}", self.save_dir)

            # Sum over the last three dimensions to get the processed segment of patches
            patch = patch.sum(dim=(5, 6, 7)).reshape(bs, s, s, oc)
            out[:, hi * s : (hi + 1) * s, wi * s : (wi + 1) * s].copy_(patch)

        if module.bias is not None:
            out.add_(module.bias.view(1, 1, 1, -1))

        return out.permute(0, 3, 1, 2)

    # Returns the smallest non-trivial factor of n (i.e factor > 1)
    def _get_smallest_factor(self, n):
        limit = int(n**0.5) + 1
        for i in range(2, limit):
            if n % i == 0:
                return i
        return n

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
