# ***************************************************************************************/
# *    Title: ann2snn
# *    Reference: This code is adapted from spikingJelly cnn_mnist.py
# *    Availability: https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/ann2snn/examples/cnn_mnist.py
# *    Date: 07/11/2024
# *    Code version: 0.0.0.014
# *
# ***************************************************************************************/
from copy import copy, deepcopy
import logging
from chop.ir.graph.mase_metadata import MaseMetadata
from chop.nn.snn.modules.neuron import IFNode
from chop.passes.graph.transforms.quantize.quant_parsers.update_node_meta import (
    update_quant_meta_param,
)
import torch
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
from chop.passes.graph.transforms.utils.conv_bn_fusion import (
    conv_bn_fusion_transform_pass,
)
from chop.nn.snn.modules.modules import VoltageHook, VoltageScaler
from tqdm import tqdm
from typing import Tuple

from ...utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

CONVERTABLE_OP = {
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "relu",
    "sub",
    "batch_norm2d",
    "layer_norm",
}


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def attach_empty_mase_metadata(node):
    node.meta["mase"] = MaseMetadata(node=node)
    return node


def add_module_and_node(
    fx_model: torch.fx.GraphModule,
    target: str,
    after: torch.fx.Node,
    m: torch.nn.Module,
    args: Tuple,
) -> torch.fx.Node:
    """Add a node m with target name after the after_node"""
    fx_model.add_submodule(target=target, m=m)
    with fx_model.graph.inserting_after(n=after):
        new_node = fx_model.graph.call_module(module_name=target, args=args)
    return new_node


def replace_by_ifnode(graph, config: dict) -> torch.fx.GraphModule:
    """
    * :ref:`API in English <Converter.replace_by_ifnode-en>`

    .. replace_by_ifnode-en:

    :param fx_model: Original fx_model
    :type fx_model: torch.fx.GraphModule
    :return: fx_model whose ReLU has been replaced by IF neuron.
    :rtype: torch.fx.GraphModule

    ``replace_by_ifnode`` is used to replace ReLU with IF neuron.

    """

    # TODO: Many of the code here need to be refactored when the spiking mase graph is available

    hook_cnt = -1
    fx_model = graph.model

    # for node in fx_model.graph.nodes:
    for node in graph.fx_graph.nodes:
        if node.op != "call_module":
            continue

        if type(fx_model.get_submodule(node.target)) is VoltageHook:
            if type(fx_model.get_submodule(node.args[0].target)) is torch.nn.ReLU:
                node_config = get_config(config, get_mase_op(node.args[0]))

                hook_cnt += 1
                hook_node = node
                relu_node = node.args[0]
                if len(relu_node.args) != 1:
                    raise NotImplementedError(
                        "The number of relu_node.args should be 1."
                    )
                s = fx_model.get_submodule(node.target).scale.item()
                target0 = "snn tailor." + str(hook_cnt) + ".0"  # voltage_scaler
                target1 = "snn tailor." + str(hook_cnt) + ".1"  # IF_node
                target2 = "snn tailor." + str(hook_cnt) + ".2"  # voltage_scaler
                m0 = VoltageScaler(1.0 / s)
                if node_config.get("name") == "IFNode":
                    m1 = IFNode(v_threshold=1.0, v_reset=None)
                else:
                    raise NotImplementedError("Not implemented yet.")
                m2 = VoltageScaler(s)

                node0 = add_module_and_node(
                    fx_model, target0, hook_node, m0, relu_node.args
                )
                node0 = attach_empty_mase_metadata(node0)

                # parent_name, name = get_parent_name(node.target)
                # setattr(graph.modules[parent_name], name, m0)

                node1 = add_module_and_node(fx_model, target1, node0, m1, (node0,))
                node1 = attach_empty_mase_metadata(node1)

                node2 = add_module_and_node(fx_model, target2, node1, m2, args=(node1,))
                node2 = attach_empty_mase_metadata(node2)

                relu_node.replace_all_uses_with(node2)
                node2.args = (node1,)
                fx_model.graph.erase_node(hook_node)
                fx_model.graph.erase_node(relu_node)
                fx_model.delete_all_unused_submodules()
    fx_model.graph.lint()
    fx_model.recompile()

    return graph.model


def graph_iterator_ann2snn_by_name(graph, config: dict):
    pass


def graph_iterator_ann2snn_by_type(graph, config: dict):
    fuse_flag = config.get("fuse", False)
    dataloader = config.get("train_data_loader")
    device = config.get("device", "cpu")

    if fuse_flag:
        graph, _ = conv_bn_fusion_transform_pass(graph)

    hook_cnt = -1

    # Adding hooks to the graph
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].parameters["common"] == {}:
            # spiking node! Ignore for now
            continue
        node_config = get_config(config, get_mase_op(node))

        if node.op == "call_module":
            # NOTE: if the following list continues to grow, consider moving it to a separate file
            if get_mase_op(node) == "relu":
                hook_cnt += 1
                target = "snn tailor." + str(hook_cnt) + ".0"  # voltage_hook]

                mode = node_config.get("mode", "99.9%")
                momentum = node_config.get("momentum", 0.1)
                m = VoltageHook(momentum=momentum, mode=mode)
                # TODO: check this
                new_node = add_module_and_node(graph.model, target, node, m, (node,))
                new_node = attach_empty_mase_metadata(new_node)

    graph.fx_graph.lint()
    graph.model.recompile()  # TODO: is this necessary?

    # calibrate the scale
    for _, imgs in enumerate(tqdm(dataloader)):
        graph.model(imgs["x"].to(device))

    # snn = replace_by_ifnode(ann_with_hook).to(self.device)
    graph.model = replace_by_ifnode(graph, config).to(device)

    return graph  # return type: GraphModule


def ann2snn_transform_pass(graph, pass_args=None):
    """
    Transform the graph from ANN to SNN.

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    .. code-block: python

        quan_args = {
            "by": "type", # quantize by type, name, or regex_name
            "default": {"config": {"name": None}}, # default config, this would be used for any node that does not have a specific config
            "relu": {
                "config": {
                    "name": "IFNode",  # conversion scheme name supported are ["IFNode", "LIFNode"...]

                    # Voltage normalization (ensure the output of the activation function is within the range of the neuron model [0,1])
                    "mode": "99.9%", # conversion mode supported are ["max", "99.9%", 1.0/2, 1.0/3. 1.0/4, 1.0/5]
                    "momentum": 0.1, # momentum for the voltage normalization
                    "fuse": True, # Bool if true: fusing the conv and bn layer, vice versa
                    "device": "cpu", # device to perform the calibration
                }
            },
        }

    :return: The transformed graph.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.

    """
    by = pass_args.pop("by")
    match by:
        case "type":
            graph = graph_iterator_ann2snn_by_type(graph, pass_args)
        case "name":
            graph = graph_iterator_ann2snn_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}
