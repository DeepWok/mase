"""
Fused Add + RMSNorm Transform Pass for MASE
============================================

This pass walks a MaseGraph's FX graph, pattern-matches the sequence:

    residual = residual + hidden_states          (an `add` node)
    normed   = rmsnorm(residual, weight)         (a `call_module` targeting an RMSNorm)

and replaces both nodes with a single call to FusedAddRMSNormModule,
which invokes a hand-written Triton kernel that fuses the two operations
into a single GPU kernel launch.

Usage within the MASE pipeline:

    from chop import MaseGraph
    from chop.passes.graph.transforms.fused_rmsnorm import fused_rmsnorm_transform_pass

    mg = MaseGraph(model)
    mg, _ = fused_rmsnorm_transform_pass(mg, pass_args={
        "casting_mode": "llama",   # "llama", "gemma", or "none"
    })

Part 2 of the ADLS kernel-fusion-aware optimisation pipeline.

Author : ADLS Group (Software Stream)
Date   : March 2026
"""

import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

try:
    from torch.fx import Node
except ImportError:
    Node = None

from .triton_fused_add_rmsnorm import FusedAddRMSNormModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm class names we recognise as pattern targets
# ---------------------------------------------------------------------------
# HuggingFace models use different class names depending on the model family.
# We match any module whose class name contains one of these substrings.
_RMSNORM_CLASS_NAMES = frozenset({
    "RMSNorm",
    "LlamaRMSNorm",
    "MistralRMSNorm",
    "GemmaRMSNorm",
    "Qwen2RMSNorm",
    "InternLMRMSNorm",
    "CohereLayerNorm",  # Cohere uses RMSNorm-style norm with a different name
})


def _is_rmsnorm_module(module: nn.Module) -> bool:
    """Check if a module is an RMSNorm variant by class name."""
    cls_name = type(module).__name__
    return any(name in cls_name for name in _RMSNORM_CLASS_NAMES)


def _get_rmsnorm_params(module: nn.Module) -> dict:
    """
    Extract the hidden_size, eps, and weight offset from a recognised
    RMSNorm module.  Different HuggingFace model families store these
    under slightly different attribute names — this helper normalises them.
    """
    # Hidden size — all known RMSNorm variants store the weight as a 1-D param
    if hasattr(module, "weight"):
        hidden_size = module.weight.shape[0]
    else:
        raise ValueError(
            f"Cannot extract hidden_size from {type(module).__name__}: "
            "module has no `weight` attribute."
        )

    # Epsilon
    eps = getattr(module, "variance_epsilon", None)  # Llama, Mistral
    if eps is None:
        eps = getattr(module, "eps", 1e-6)  # Gemma, generic

    # Weight offset (Gemma adds 1.0 to the weight)
    cls_name = type(module).__name__
    if "Gemma" in cls_name:
        offset = 1.0
    else:
        offset = 0.0

    return {
        "hidden_size": hidden_size,
        "eps": eps,
        "offset": offset,
    }


def _is_add_node(node: "Node") -> bool:
    """
    Return True if an FX node represents a tensor addition.

    In traced FX graphs, addition can appear as:
      - call_function targeting operator.add or torch.add
      - call_method with target "add"
    """
    import operator

    if node.op == "call_function":
        return node.target in (operator.add, torch.add)
    if node.op == "call_method":
        return node.target == "add"
    return False


def _is_rmsnorm_node(node: "Node", graph_module: nn.Module) -> bool:
    """
    Return True if an FX node is a call_module targeting a recognised
    RMSNorm variant.
    """
    if node.op != "call_module":
        return False
    try:
        target_module = graph_module.get_submodule(node.target)
    except AttributeError:
        return False
    return _is_rmsnorm_module(target_module)


# ---------------------------------------------------------------------------
# Core pattern matching: find (add, rmsnorm) pairs
# ---------------------------------------------------------------------------
def _find_add_rmsnorm_pairs(
    graph_module: nn.Module,
) -> list:
    """
    Walk the FX graph and return a list of (add_node, rmsnorm_node) tuples
    where the rmsnorm_node consumes the output of the add_node as its first
    positional argument, and the add_node has no other consumers besides the
    rmsnorm_node (and possibly a downstream residual consumer — which we
    handle by emitting the residual output from the fused module).

    We also accept the case where the add result flows through the rmsnorm
    AND through other downstream nodes (the residual stream).  The fused
    module produces both outputs, so we can rewire.
    """
    fx_graph = graph_module.graph
    pairs = []

    for node in fx_graph.nodes:
        # Step 1: is this node an RMSNorm call?
        if not _is_rmsnorm_node(node, graph_module):
            continue

        # Step 2: is the first argument to this RMSNorm an add node?
        if len(node.args) == 0:
            continue
        maybe_add = node.args[0]
        if not isinstance(maybe_add, Node):
            continue
        if not _is_add_node(maybe_add):
            continue

        pairs.append((maybe_add, node))

    return pairs


# ---------------------------------------------------------------------------
# Graph rewriting
# ---------------------------------------------------------------------------
def _replace_pair(
    graph_module: nn.Module,
    add_node: "Node",
    rmsnorm_node: "Node",
    casting_mode: str,
    fused_module_counter: int,
) -> int:
    """
    Replace a matched (add, rmsnorm) pair with a FusedAddRMSNormModule.

    Returns the updated counter for naming.
    """
    fx_graph = graph_module.graph

    # ---- 1. Read params from the original RMSNorm module ----
    orig_rmsnorm = graph_module.get_submodule(rmsnorm_node.target)
    params = _get_rmsnorm_params(orig_rmsnorm)

    # ---- 2. Create the fused module ----
    fused_mod = FusedAddRMSNormModule(
        hidden_size=params["hidden_size"],
        eps=params["eps"],
        offset=params["offset"],
        casting_mode=casting_mode,
    )

    # Copy the learned weight from the original RMSNorm
    with torch.no_grad():
        fused_mod.weight.copy_(orig_rmsnorm.weight)

    # ---- 3. Register the fused module in the graph_module hierarchy ----
    fused_name = f"fused_add_rmsnorm_{fused_module_counter}"
    graph_module.add_module(fused_name, fused_mod)

    # ---- 4. Insert a call_module node for the fused op ----
    # The add node has two inputs: the residual and the hidden states.
    # Recover them from the add node's args.
    add_args = add_node.args
    if len(add_args) >= 2:
        x_residual = add_args[0]
        x_hidden = add_args[1]
    else:
        # Fallback for call_method style: self.add(other)
        x_residual = add_args[0] if len(add_args) > 0 else add_node
        x_hidden = add_node.kwargs.get("other", add_args[1] if len(add_args) > 1 else None)

    # Insert the new node right after the rmsnorm node
    with fx_graph.inserting_after(rmsnorm_node):
        fused_node = fx_graph.call_module(
            fused_name,
            args=(x_residual, x_hidden),
        )

    # The fused module returns (normed_out, residual_out) as a tuple.
    # We need getitem nodes to unpack.
    with fx_graph.inserting_after(fused_node):
        normed_getitem = fx_graph.call_function(
            target=lambda tup, idx: tup[idx],
            args=(fused_node, 0),
        )
        # Use operator.getitem for clean FX graph
        import operator
        normed_getitem.target = operator.getitem
        normed_getitem.args = (fused_node, 0)

    with fx_graph.inserting_after(normed_getitem):
        residual_getitem = fx_graph.call_function(
            target=lambda tup, idx: tup[idx],
            args=(fused_node, 1),
        )
        residual_getitem.target = operator.getitem
        residual_getitem.args = (fused_node, 1)

    # ---- 5. Rewire consumers ----
    # All consumers of the original rmsnorm_node now consume normed_getitem
    rmsnorm_node.replace_all_uses_with(normed_getitem)
    # Fix self-reference: normed_getitem's arg should point to fused_node, not itself
    normed_getitem.args = (fused_node, 0)

    # All consumers of the original add_node (other than the rmsnorm) now
    # consume residual_getitem.  This handles the residual stream.
    add_node.replace_all_uses_with(residual_getitem)
    # Fix: the fused_node's args still need the original inputs, not residual_getitem
    fused_node.args = (x_residual, x_hidden)
    # Fix: residual_getitem's arg should point to fused_node
    residual_getitem.args = (fused_node, 1)

    # ---- 6. Erase the old nodes (rmsnorm first, then add — order matters) ----
    fx_graph.erase_node(rmsnorm_node)
    fx_graph.erase_node(add_node)

    logger.info(
        f"Fused add + RMSNorm: {add_node.name} + {rmsnorm_node.name} "
        f"-> {fused_name} (casting_mode={casting_mode})"
    )

    return fused_module_counter + 1


# ---------------------------------------------------------------------------
# Public pass function (MASE convention: takes graph, returns (graph, {}))
# ---------------------------------------------------------------------------
def fused_rmsnorm_transform_pass(
    graph,
    pass_args: Optional[Dict[str, Any]] = None,
) -> Tuple:
    """
    Apply fused add + RMSNorm transformation to the given MaseGraph.

    This pass walks the FX graph, identifies patterns where a tensor
    addition is immediately followed by an RMSNorm call, and replaces
    both with a single FusedAddRMSNormModule backed by an optimised
    Triton kernel.

    Parameters
    ----------
    graph : MaseGraph
        The input graph to be transformed.
    pass_args : dict, optional
        Configuration for the pass:
            - casting_mode (str): "llama" (default), "gemma", or "none".
              Controls numerical precision during normalisation.

    Returns
    -------
    tuple
        (transformed_graph, info_dict) following MASE pass convention.

    Example
    -------
    >>> from chop import MaseGraph
    >>> mg = MaseGraph(model)
    >>> mg, info = fused_rmsnorm_transform_pass(mg, {"casting_mode": "llama"})
    """
    pass_args = pass_args or {}
    casting_mode = pass_args.get("casting_mode", "llama")

    # MaseGraph wraps a torch.fx.GraphModule — get it
    graph_module = graph.model if hasattr(graph, "model") else graph

    # Find all (add, rmsnorm) pairs
    pairs = _find_add_rmsnorm_pairs(graph_module)

    if not pairs:
        logger.info("fused_rmsnorm_transform_pass: no add+RMSNorm patterns found.")
        return graph, {}

    logger.info(
        f"fused_rmsnorm_transform_pass: found {len(pairs)} add+RMSNorm "
        f"pattern(s) to fuse."
    )

    # Replace each pair
    counter = 0
    for add_node, rmsnorm_node in pairs:
        counter = _replace_pair(
            graph_module, add_node, rmsnorm_node, casting_mode, counter
        )

    # Recompile the graph after mutations
    graph_module.graph.lint()
    graph_module.recompile()

    return graph, {"num_fused": counter}
