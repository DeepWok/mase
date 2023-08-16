"""
The entrypoint to chop's pruning transforms

The pruning passes are public-facing entrypoints to the functionality exposed by the 
pruning transform. Currently, we support a layerwise pruning strategy for weights and
support activation pruning as an add-on. Please take a look at the methods.py file for
more details on the pruning methods. The iterator steps through compatible nodes in the
graph, wraps them with an activation pruning handler (if applicable), and prunes them.

NOTE: We only support vision models targeted for classification tasks. Also, we
currently only support one-shot, locally-scoped unstructured pruning; we don't support
fine-tuning with pruning yet.

Example Usage:
# For example configurations, see the configs/tests/prune directory
$ python tests/passes/transforms/prune/toy.py
$ python tests/passes/transforms/prune/prune.py <config_name>
(or)
$ ./ch transform --config configs/tests/prune/resnet18_fixed_global.toml --pretrained

----------------------------------------------------------------------------------------
Internal Backlog:
1. Add support for fine-tuning with pruning via registering weight masks via the wrap
   function of the pruner. The unwrap pass would then take the responsibility of
   removing these masks to make the pruning permanent. The exact specifics may require
   a bit of thought as the model will have to be saved and then loaded again for
   training. Perhaps we could create a wrapper class (nn.Module) that uses a custom
   forward function to apply the masks while training? 
2. Currently, our calibration pass only uses a single batch to generate a report on the
   channel-wise activation sparsity. Perhaps we could aggregate statistics over multiple
   batches to get a better estimate.
3. Add tests for the weight and activation configuration validation functions.
4. Consider extending PyTorch's built-in pruning module to support activation sparsity;
   this is an attractive option as it would allow us to leverage the existing pruning
   workflows (incl. globally unstructured ones).
"""

import torch
import torch.nn as nn
from tqdm.contrib.logging import tqdm_logging_redirect

from chop.passes.graph.mase_graph import MaseGraph
from chop.passes.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.tools.logger import getLogger

from .criteria import RANK_CRITERIA
from .methods import LevelPruner, ACTIVATION_PRUNE_STRATEGIES, ActivationPruneHandler

# Constants ----------------------------------------------------------------------------
# Pruneable MASE operators
# NOTE: We don't do activation pruning for conv1d and linear layers.
PRUNEABLE_OPS = {"conv1d": nn.Conv1d, "conv2d": nn.Conv2d, "linear": nn.Linear}

# A registry of available pruning strategies (i.e. algorithms)
PRUNE_METHODS = {
    # A basic one-shot pruner that prunes to a given sparsity level
    "level-pruner": LevelPruner,
    # Add more here...
}

# Housekeeping -------------------------------------------------------------------------
logger = getLogger(__file__)
logger.propagate = False  # Avoid duplicate logs


# Passes -------------------------------------------------------------------------------
# NOTE: This is where you'd handle layer-specific pruning (i.e. by name, type, etc.)
# by calling different graph iterators. For now, we stick with a simple iterator
# that prunes all supported operators.
# See machop/chop/passes/transforms/quantize/quantize.py for an example.
def prune_transform_pass(graph: MaseGraph, save_dir=None, config=None):
    return prune_graph_iterator(graph, save_dir, config)


# NOTE: For now, this pass takes care of unwrapping the activation prune handler, but
# later this will be useful for making pruning permanent by removing masks.
def prune_unwrap_transform_pass(graph: MaseGraph, *_):
    # Remove the activation hooks and the handler from the model
    handler = getattr(graph.model, "activation_prune_handler", None)
    if handler:
        handler.unwrap()
        delattr(graph.model, "activation_prune_handler")
        logger.info("Removed activation pruning hooks and handler from the model")
    else:
        logger.info(
            "No activation handler found in the model. Did you make sure to load the "
            "model from a mase graph module (.mz) file? Skipping unwrap."
        )
    return graph


# Iterators ----------------------------------------------------------------------------
def prune_graph_iterator(graph: MaseGraph, save_dir: str, config: dict):
    # This generator is used to create a batch for a sample forward pass.
    input_generator = config.get("input_generator", None)

    # Setup all pruning-related parameters (incl. basic validation)
    method, criterion, sparsity = get_weight_prune_params(config["weight"])
    prune_activations = "activation" in config
    strategy, target = (
        get_activation_prune_params(config["activation"], graph)
        if prune_activations
        else (None, None)
    )

    handler = ActivationPruneHandler(strategy, target) if prune_activations else None
    if handler:
        # Register the handler as an attribute of the model; since we're using a partial
        # function for the pre-forward hook and the handler is passed in as an argument,
        # we'd lose the reference when saving the graph if we didn't do this.
        setattr(graph.model, "activation_prune_handler", handler)
        logger.debug("Added the activation prune handler as a model attribute")
    pruner = method(sparsity, criterion, handler)

    # Iterate over the graph and prune the compatible nodes
    total = len(graph.fx_graph.nodes)
    with tqdm_logging_redirect(total=total, loggers=[logger]) as pbar:
        for node in graph.fx_graph.nodes:
            if get_mase_op(node) in ["batch_norm1d", "batch_norm2d"]:
                logger.warning(
                    f"Batchnorm layer detected ({node.target}). This could corrupt "
                    "sparsity in the weights when fused later."
                )

            if get_mase_op(node) not in PRUNEABLE_OPS.keys():
                # Skip if the operator is not supported for pruning
                pbar.update(1)
                continue

            if get_mase_type(node) == "module_related_func":
                module = get_node_actual_target(node)

                # NOTE: This is where you'd prepare keyword args for a custom criterion
                kwargs = {}

                # Wrap and prune the module :)
                pbar.set_description(f"Pruning candidate node {node.target}")
                pruner.wrap(module, node.target)
                if handler:
                    handler.wrap(module, node.target)
                pruner.apply(module, node.target, **kwargs)

            pbar.update(1)
        pbar.set_description("Done")

    # A sample forward pass to log activation sparsity statistics :)
    # NOTE: Currently, we only do this on one batch sample. As in line 230 of
    # machop/chop/passes/analysis/statistical_profiler/profile_statistics.py,
    # we may want to aggregate statistics over multiple batches.
    # ----------------------------------------------------------------------------------
    # Also, as mentioned earlier,
    if handler:
        logger.info("Running inference to generate an activation sparsity report")
        previous_state = graph.model.training
        graph.model.train(False)
        with torch.inference_mode():
            batch = next(input_generator)["x"]
            graph.model(batch)
        graph.model.train(previous_state)

    # Summary
    logger.info("Weight Pruning Summary:")
    pruner.show_summary()
    if prune_activations:
        logger.info("Activation Pruning Summary:")
        handler.show_summary()

    # Save the pruning report and summary
    pruner.save_summary(save_dir)
    if handler:
        handler.save_report(save_dir)
        handler.save_summary(save_dir)

    return graph


# Helper functions ---------------------------------------------------------------------
# TODO: We need to add tests for these functions with a variety of configurations.
def get_weight_prune_params(config: dict):
    method = config.get("method", "level-pruner")
    criterion = config.get("criterion", "random")
    sparsity = config.get("sparsity", 0.0)

    # Validate the parameters
    if method not in PRUNE_METHODS.keys():
        raise ValueError(
            "Unsupported pruning method {}. Please choose from {}".format(
                method, list(PRUNE_METHODS.keys())
            )
        )
    if criterion not in RANK_CRITERIA.keys():
        raise ValueError(
            "Unsupported ranking criterion {}. Please choose from {}".format(
                criterion, list(RANK_CRITERIA.keys())
            )
        )
    if not isinstance(sparsity, float):
        raise ValueError("Sparsity must be a float. Got {}".format(type(sparsity)))
    if sparsity < 0 or sparsity > 1:
        raise ValueError("Sparsity must be between 0 and 1. Got {}".format(sparsity))

    method = PRUNE_METHODS[method]
    criterion = RANK_CRITERIA[criterion]

    return method, criterion, sparsity


def get_activation_prune_params(config: dict, graph: MaseGraph):
    strategy = config.get("strategy", "fixed-global")
    target = config.get("target", 0.1)

    # Validate the parameters
    if strategy not in ACTIVATION_PRUNE_STRATEGIES:
        raise ValueError(
            "Unsupported activation pruning strategy {}. Please choose from {}".format(
                strategy, list(ACTIVATION_PRUNE_STRATEGIES)
            )
        )
    # Validate the target based on the choice of strategy
    # NOTE: In the observe strategy, the target isn't used, so we don't care about it.
    if strategy == ACTIVATION_PRUNE_STRATEGIES[3]:  # fixed-layerwise
        # Verify that the target's a dictionary and that its keys are valid node names
        if not isinstance(target, dict):
            raise ValueError(
                "Expected dictionary of layerwise thresholds. Got {}".format(
                    type(target)
                )
            )
        names = [node.target for node in graph.fx_graph.nodes]  # Names of all nodes
        for name in target.keys():
            if name not in names:
                raise ValueError(f"Node {name} not found in the graph!")
    else:
        # Verify that the target's a float
        if not isinstance(target, float):
            raise ValueError("Target must be a float. Got {}".format(type(target)))
        # Verify that the target's between 0 and 1 if the strategy is adaptive
        # NOTE: Here, the target is treated like sparsity whereas in both the other
        # cases, it's treated like a threshold.
        if strategy == ACTIVATION_PRUNE_STRATEGIES[1] and (target < 0 or target > 1):
            raise ValueError("Target must be between 0 and 1. Got {}".format(target))

    return strategy, target
