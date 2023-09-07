# A pass to fuse batch normalisation layers with the preceeding convolutional layers
# NOTE: This implementation is a derivative of the following:
# https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/optimization.py

import torch.nn as nn
import torch.fx as fx
from tqdm.contrib.logging import tqdm_logging_redirect
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)

from chop.tools.logger import getLogger

# Housekeeping -------------------------------------------------------------------------
logger = getLogger(__file__)
logger.propagate = False  # Avoid duplicate logs


def conv_bn_fusion_transform_pass(graph, **_):
    PATTERNS = [
        (nn.Conv1d, nn.BatchNorm1d),
        (nn.Conv2d, nn.BatchNorm2d),
        (nn.Conv3d, nn.BatchNorm3d),
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
                        logger.warning("Conv output used by other nodes. Skipped!")
                        continue
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    if not bn.track_running_stats:
                        # When track_running_stats is False, the batch norm module's
                        # running mean and variance buffers are set to None.
                        logger.warning("Batchnorm not tracking stats. Skipped!")
                        continue
                    # Set both modules to eval mode
                    conv_prev_mode, bn_prev_mode = conv.training, bn.training
                    conv.train(False)
                    bn.train(False)
                    # Fuse!
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    # Restore the previous modes
                    conv.train(conv_prev_mode)
                    bn.train(bn_prev_mode)
                    # NOTE: We may need to update metadata here. Currently unclear.
                    # Replace conv with the fused module and erase the batchnorm node
                    replace_node_module(node.args[0], modules, fused_conv)
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
