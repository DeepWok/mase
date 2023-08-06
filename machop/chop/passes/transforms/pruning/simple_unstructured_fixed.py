"""
Pruning passes 

NOTE: The code here is currently under development. Feel free to browse the pruning 
roadmap at the link below and flag any issues you may find: 
https://github.com/JianyiCheng/mase-tools/issues/314
----------------------------------------------------------------------------------------
Internal Backlog:
1. Push constants to a separate file
2. Refactor the pruning workflow to split it into multiple passes, all defined in
   separate files
3. Consider extending PyTorch's built-in pruning module to support activation sparsity;
   this is an attractive option as it would allow us to leverage the existing pruning
   workflows (incl. globally unstructured ones).
4. Update the naming of the adaptive pruning mode to reflect the fact that the threshold
   passed in will be treated as a percentage. 
5. Add support for a layerwise activation threshold configuration via the config file.
   The thresholds could be computed on some representative sample of a target dataset
   using a separate calibration pass or via the statistics profiler. This allows good
   flexibility in the pruning workflow.
6. Setup a globally accessible project log directory variable. 
7. Add a check to make sure that the sparsity threshold is between 0 and 1.
8. More listed below in context.
"""

from collections import OrderedDict
from pathlib import Path

import torch
from tabulate import tabulate
from chop.passes.utils import get_mase_op, get_mase_type, get_node_actual_target

# Constants ----------------------------------------------------------------------------
PRUNEABLE_OP = ["conv1d", "conv2d", "linear"]
PRUNING_MODE = ["l1-unstructured", "random-unstructured"]
PRUNING_TARGET = ["weight", "activation", "both"]


# Helper functions ---------------------------------------------------------------------
# NOTE: The workflow here is quite similar to that in PyTorch's built-in pruning module.
def apply_parameter_pruning(module, name, mask):
    # Re-register the original weight under a different name and delete its attribute
    original_weight = getattr(module, name)
    module.register_parameter(f"{name}_original", original_weight)
    del module._parameters[name]

    # Apply the mask to the weight and store it under the original name
    module.register_buffer(f"{name}_mask", mask)
    setattr(module, name, original_weight * mask)


# NOTE: There may be an obscure case where the user wants to remove the handles for a
# particular module. This is currently not supported, but can be by converting the
# list into an ordered dictionary.
class ActivationPruner:
    def __init__(self):
        self.table = []
        self.activations = OrderedDict()
        self.activation_masks = OrderedDict()
        self.handles = []

    def __call__(self, name, threshold, adaptive=False, save=False):
        def pre_forward_hook(_, args):
            input = args[0]

            activation_threshold = (
                torch.quantile(input.abs().flatten(), threshold)
                if adaptive
                else threshold
            )

            # Use the activation threshold to create a mask on the input
            activation_mask = input.abs() > activation_threshold
            activation_mask = activation_mask.to(dtype=input.dtype)

            if save:
                self.activation_masks[name] = activation_mask

            # Log basic pruning statistics to a table :)
            tot_elements = input.numel()
            nnz_elements_old = torch.count_nonzero(input)
            nnz_elements_new = torch.count_nonzero(input * activation_mask)
            sparsity_old = 1 - nnz_elements_old / tot_elements
            sparsity_new = 1 - nnz_elements_new / tot_elements
            self.table.append(
                [
                    name,
                    f"{sparsity_old:.4f}",
                    f"{sparsity_new:.4f}",
                    f"{tot_elements:,}",
                    f"{nnz_elements_old:,}",
                    f"{nnz_elements_new:,}",
                    f"({', '.join(map(str, list(input.shape)))})",
                    f"{activation_threshold:.4f}",
                    f"{torch.min(input):.4f}",
                    f"{torch.max(input):.4f}",
                ]
            )

            # Apply the mask to the input
            return input * activation_mask

        def forward_hook(module, input, output):
            # Store the output of the module
            self.activations[name] = output.detach()

        return pre_forward_hook, forward_hook

    def summary(self):
        fields = [
            "Layer",
            "Sparsity (Old)",
            "Sparsity (New)",
            "Total",
            "NNZ (Old)",
            "NNZ (New)",
            "Shape",
            "Threshold",
            "Min",
            "Max",
        ]
        print(tabulate(self.table, headers=fields, tablefmt="pretty"))


# Pruning strategies -------------------------------------------------------------------
# NOTE: Under construction...
# Pre: Pruning type is validated
def simple_unstructured_fixed_pruning(
    module: torch.nn.Module,
    pruning_type: str,
    sparsity: float,
    activation_threshold: float,  # Currently unused
):
    # Weight pruning
    if pruning_type in ["weight", "both"]:
        weight = module.weight.data
        weight_threshold = torch.quantile(weight.abs().flatten(), sparsity)
        weight_mask = weight.abs() > weight_threshold
        weight_mask = weight_mask.to(dtype=weight.dtype)
        apply_parameter_pruning(module, "weight", weight_mask)

    # Activation pruning
    if pruning_type in ["activation", "both"]:
        # TODO: To be implemented
        ...


# NOTE: Under the adaptive mode, the "activation_threshold" acts as a percentage, and
# the real threshold is computed using torch.quantile on the flattened map. The naming
# is quite confusing and should be changed soon.
def random_unstructured_pruning(
    module: torch.nn.Module,
    key: str,
    adaptive: bool,
    sparsity: float,
    activation_threshold: float,
    pruner: ActivationPruner,
):
    # Weight pruning
    weight = module.weight.data
    weight_mask = torch.ones(weight.size(), dtype=weight.dtype)

    # Set sparsity percentage of values in the mask to 0 randomly
    weight_mask[torch.rand(weight.size()) < sparsity] = 0
    apply_parameter_pruning(module, "weight", weight_mask)

    # Activation pruning
    pre_forward_hook, forward_hook = pruner(key, activation_threshold, adaptive)
    pruner.handles.append(module.register_forward_pre_hook(pre_forward_hook))
    pruner.handles.append(module.register_forward_hook(forward_hook))


# Pruning handlers ---------------------------------------------------------------------
# NOTE: This function needs to be refactored as it's handling way more than it should.
# Ideally, the pruning routine should be split into compartmentalized functions,
# possibly even a list of callbacks to a minimal core routine. It's also a must to
# document the required parameters in the pruning config; a separate validation pass
# over the configuration file would be ideal.
def graph_iterator_prune(graph, config: dict):
    # Housekeeping on the pruning mode and type
    # NOTE: Currently, only the random unstructured pruning mode is fully supported.
    mode = config.get("mode", PRUNING_MODE[0])
    if mode not in PRUNING_MODE:
        raise ValueError(
            f"Invalid pruning mode {mode}. Valid modes are: {', '.join(PRUNING_MODE)}"
        )
    target = config.get("target", PRUNING_TARGET[0])
    if target not in PRUNING_TARGET:
        raise ValueError(
            "Invalid pruning type {}. Valid types are: {}".format(
                target, ", ".join(PRUNING_TARGET)
            )
        )
    # NOTE: Currently, random unstructured pruning enforces "both"
    target = target if mode == PRUNING_MODE[0] else PRUNING_TARGET[2]

    table = []
    dummy_input = None
    weights, weight_masks = OrderedDict(), OrderedDict()
    activation_pruner = None if target == PRUNING_TARGET[0] else ActivationPruner()

    # A toggle to save weight and/or activation masks, intermediary activations and the
    # model weights as a dictionary indexed by the node's target name.
    # NOTE: Maybe later, allow users flexibility in specifying exactly what to save. If
    # the user specifies a pruning mode of only "activation", the weight masks and
    # weights aren't stored (and vice versa). Again, it'd be good to offer control over
    # this in the future.
    save = config.get("save", False)
    save_dir = config.get("save_dir", None)
    if save:
        # Make sure that the path exists and points to a directory
        save_dir = Path(save_dir, exist_ok=True, is_dir=True).absolute()

    # Iterate over the graph and prune the compatible nodes
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in PRUNEABLE_OP:
            # Skip if the operator is not supported for pruning
            continue

        if get_mase_type(node) == "module_related_func":
            # NOTE: As mentioned above, ideally this function should be agnostic to the
            # pruning approach. An alternative to the callback approach would be to
            # maintain a sort of pruning registry that maps a mode to a supported
            # pruning function. These functions would then independently handle the
            # parameters they require (e.g. granularity, sparsity, saliency criteria,
            # etc.). An interesting problem to tackle is that the scope for pruning
            # here is limited to the local (layer-level) scope due to the for loop over
            # compatible nodes. Ranking the weights for a globally-scoped approach would
            # requite an alternate approach.
            if mode == PRUNING_MODE[0]:
                simple_unstructured_fixed_pruning(
                    get_node_actual_target(node),
                    target,
                    config.get("sparsity", 0),  # Used for weight pruning
                    config.get("threshold", 0),  # Used for natural pruning
                )
            elif mode == PRUNING_MODE[1]:
                # The random pruning mode performs both weight and activation pruning
                # but with random masks. The saliency criteria is not used here.
                random_unstructured_pruning(
                    get_node_actual_target(node),
                    node.target,  # Key for the weight-activation mask dictionary
                    config.get("adaptive", False),
                    config.get("sparsity", 0),
                    config.get("threshold", 6),
                    activation_pruner,
                )

            if target == PRUNING_TARGET[1]:
                continue

            # Log basic pruning statistics to a table :)
            module = get_node_actual_target(node)
            tot_weights = module.weight.numel()
            nnz_weights = torch.count_nonzero(module.weight)
            sparsity = 1 - nnz_weights / tot_weights

            # Format the total and non-zero weights to be more readable with commas
            tot_weights = "{:,}".format(tot_weights)
            nnz_weights = "{:,}".format(nnz_weights)

            table.append([node.target, f"{sparsity:.4f}", tot_weights, nnz_weights])

            # Log the weight masks
            if save:
                weights[node.target] = module.weight
                weight_masks[node.target] = getattr(module, "weight_mask")

            # Remove the "weight_original" and "weight_mask" attributes
            # NOTE: I'm unsure of why we're storing this in the first place. Perhaps
            # it's to allow for iterative pruning or just convenience?
            module._parameters.pop("weight_original")
            module._buffers.pop("weight_mask")

    if mode != PRUNING_TARGET[1]:
        # Weight pruning summary
        fields = ["Layer", "Sparsity", "Total", "NNZ (Old)"]
        print(tabulate(table, headers=fields, tablefmt="pretty"))

    if mode != PRUNING_TARGET[0]:
        # Actiation pruning inference pass and summary
        input_shape = config.get("input_shape", None)
        batch_size = config.get("batch_size", 8)

        # Dummy input between 0 and 1 on the device of the module
        dummy_input = torch.rand(batch_size, *input_shape, device=torch.device("cpu"))
        with torch.inference_mode():
            graph.model(dummy_input)
        activation_pruner.summary()

        # Remove the hooks
        for handle in activation_pruner.handles:
            handle.remove()

    if save:
        # Save the masks, weights and activations to the specified directory
        # NOTE: Missing check for when the target is only "weight" and activation_pruner
        # is None; this will cause an error.
        torch.save(
            {
                "inputs": dummy_input,
                "weights": weights,
                "weight_masks": weight_masks,
                "activations": activation_pruner.activations,
                "activation_masks": activation_pruner.activation_masks,
            },
            save_dir / "info.pt",  # Needs better naming
        )

        # Save the pruned model
        torch.save(graph.model.state_dict(), save_dir / "pruned_model.pt")

    return graph


def prune_transform_pass(graph, config=None):
    # NOTE: Currently, this code only works with the test script that passes the
    # correct configuration (i.e. config["passes"]["prune"])
    graph = graph_iterator_prune(graph, config)
    return graph
