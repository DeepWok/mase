# A pass to eliminate any activation functions within the preceding LogicNets layer is made. The activation function is already considered during the initialization of the LogicNets.
# NOTE: This implementation is a derivative of the following:
# https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/optimization.py

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
from chop.tools.logger import getLogger

# Housekeeping -------------------------------------------------------------------------
logger = getLogger(__file__)
logger.propagate = False  # Avoid duplicate logs


def logicnets_fusion_transform_pass(graph, **_):
    # If this pattern is matched, remove the activation function.
    PATTERNS = [
        (LinearLogicNets, nn.ReLU),
        (LinearLogicNets, nn.Tanh),
        (LinearLogicNets, nn.BatchNorm1d),
        # (Conv2DLogicNets, nn.ReLU),
        # (Conv2DLogicNets, nn.Tanh),
        # (Conv2DLogicNets, nn.BatchNorm2d),
    ]

    modules = dict(graph.model.named_modules())
    logger.debug(f"Found {len(modules)} modules in the model:\n{list(modules.keys())}")

    # Modify the graph in place.
    total = len(graph.fx_graph.nodes) * len(PATTERNS)
    with tqdm_logging_redirect(total=total, loggers=[logger]) as pbar:
        for pattern in PATTERNS:
            fst, snd = pattern[0].__name__, pattern[1].__name__
            pbar.set_description(f"Looking for pattern {fst} -> {snd}")

            # Iterate over the graph and fuse the nodes that match the patterns
            for node in graph.fx_graph.nodes:
                if matches_module_pattern(pattern, node, modules):
                    # There may be architectures where such a pattern exits. In these
                    # cases, fusion isn't trivial. For now, we'll just skip these cases.
                    if len(node.args[0].users) > 1:
                        logger.warning("Logicnets output used by other nodes. Skipped!")
                        continue

                    logicnets = modules[node.args[0].target]
                    activation = modules[node.target]

                    # TODO: special case for batchnorm. Currently unclear.
                    if activation == nn.BatchNorm1d:
                        if not activation.track_running_stats:
                            # When track_running_stats is False, the batch norm module's
                            # running mean and variance buffers are set to None.
                            logger.warning("Batchnorm not tracking stats. Skipped!")
                            continue

                    # NOTE: We may need to update metadata here. Currently unclear.
                    # Replace conv with the fused module and erase the batchnorm node
                    replace_node_module(node.args[0], modules, logicnets)
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
