# A pass to eliminate any activation functions within the preceding LogicNets layer is made. The activation function is already considered during the initialization of the LogicNets.
# NOTE: This implementation is a derivative of the following:
# https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/optimization.py
import logging

import tqdm
import torch.nn as nn
import torch.fx as fx
from tqdm.contrib.logging import tqdm_logging_redirect
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from chop.passes.transforms.quantize.quantized_modules.linear import LinearLogicNets
from chop.passes.transforms.quantize.quantized_modules.conv2d import Conv2DLogicNets

# Housekeeping -------------------------------------------------------------------------
logger = logging.getLogger(__file__)
# logger.propagate = False  # Avoid duplicate logs


def logicnets_fusion_transform_pass(graph, pass_args, **_):
    modules = dict(graph.model.named_modules())
    logger.debug(f"Found {len(modules)} modules in the model:\n{list(modules.keys())}")

    # store the logicnets nodes
    logicnets_nodes = list(pass_args.keys())

    # store the nodes which have been merged
    nodes_to_erase = []
    for node, config in pass_args.items():
        layer_inputs = config["config"]["additional_layers_inputs"]
        layer_outputs = config["config"]["additional_layers_outputs"]
        nodes_to_erase = nodes_to_erase + layer_inputs + layer_outputs

    # Modify the graph in place.
    total = len(graph.fx_graph.nodes)
    with tqdm_logging_redirect(total=total, loggers=[logger]) as pbar:
        pbar.set_description(
            f"Fusing these nodes into LogicNets layers {nodes_to_erase}"
        )

        # Iterate over the graph and erase the nodes which have been merged into a LogicNets layer
        for node in graph.fx_graph.nodes:
            if node.name in logicnets_nodes:
                # set the LogicNets nodes to 'fused' so they will apply the merged modules internally in the forward pass
                assert isinstance(
                    modules[node.target], LinearLogicNets
                ), f"{node} is not a LinearLogicNets module. Double check your model and the config file."
                modules[node.target].set_fused(True)
                # recalculate truth tables after
                modules[node.target].calculate_truth_tables()

            elif node.name in nodes_to_erase:
                # erase the nodes which have been merged into a LogicNets layer

                # There may be architectures where such a pattern exits. In these
                # cases, fusion isn't trivial. For now, we'll just skip these cases.
                if len(node.args[0].users) > 1:
                    logger.warning("Logicnets output used by other nodes. Skipped!")
                    continue

                # this is the make sure the node to erase no longer exists in the graph
                node.replace_all_uses_with(node.args[0])
                graph.fx_graph.erase_node(node)
                pbar.update(1)  # Account for removed node :)
            pbar.update(1)

        # Update the model to reflect the changes in the graph
        graph.model = fx.GraphModule(graph.model, graph.fx_graph)
        pbar.set_description("Done")

    modules = dict(graph.model.named_modules())
    logger.debug(f"Found {len(modules)} modules in the model:\n{list(modules.keys())}")
    return graph
