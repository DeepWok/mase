import logging
import math
from typing import Any

import numpy as np
import toml
import torch
from torch.fx import Interpreter
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from tqdm import tqdm

from ...utils import get_mase_op, get_mase_type, get_module_by_target
from .stat import _StatBase, create_new_stat
from .utils import get_meta_arg_stat, set_meta_arg_stat

logger = logging.getLogger(__name__)


class ActStatCollection:
    def __init__(self, stats: dict[str, dict]) -> None:
        self.stats: list[_StatBase] = []
        for stat_name, stat_config in stats.items():
            self.stats.append(create_new_stat(stat_name, **stat_config))

    def update(self, batch: torch.Tensor):
        assert isinstance(batch, torch.Tensor)
        for stat in self.stats:
            if hasattr(stat, "update_a_batch"):
                stat.update_a_batch(batch)
            else:
                for i in range(batch.size(0)):
                    stat.update_a_sample(batch[[i], ...])

    def compute(self) -> dict[str, dict[str, list]]:
        results = {}
        for stat in self.stats:
            results.update(stat.export())

        return results

    def __repr__(self) -> str:
        return "ActStatCollection(stats={})".format(
            ", ".join([type(stat).__name__ for stat in self.stats])
        )


class WeightStatCollection:
    def __init__(self, stats: dict[str, dict]) -> None:
        self.stats: list[_StatBase] = []
        for stat_name, stat_config in stats.items():
            self.stats.append(create_new_stat(stat_name, **stat_config))

    def update(self, weight: torch.Tensor):
        assert isinstance(weight, torch.Tensor)
        for stat in self.stats:
            stat: _StatBase
            stat.update_a_sample(weight)

    def compute(self) -> dict[str, dict[str, list]]:
        results = {}
        for stat in self.stats:
            results.update(stat.export())

        return results

    def __repr__(self) -> str:
        return "WeightStatCollection(stats={})".format(
            ", ".join([type(stat).__name__ for stat in self.stats])
        )


def graph_iterator_register_stat_collections_by_name(
    graph,
    target_weight_nodes: list[str],
    target_act_nodes: list[str],
    weight_stats: dict[str, dict],
    act_stats: dict[str, dict],
    profile_output_act: bool = False,
):
    # weight stats
    for node in graph.fx_graph.nodes:
        if node.name not in target_weight_nodes:
            continue
        if node.op != "call_module":
            logger.warning(
                f"Node {node.name} is not a call_module node, but is in target_weight_nodes. Skip."
            )
            continue

        # only create registered param/buffer stats for nn.Module
        for entry, s_meta in node.meta["mase"].parameters["software"]["args"].items():
            stat = s_meta["stat"]
            if "data_in" in entry:
                continue
            if isinstance(stat, (WeightStatCollection,)):
                continue
            set_meta_arg_stat(node, entry, WeightStatCollection(weight_stats))

    # act stats
    for node in graph.fx_graph.nodes:
        if node.name not in target_act_nodes:
            continue
        for entry, s_meta in node.meta["mase"].parameters["software"]["args"].items():
            stat = s_meta["stat"]
            if isinstance(stat, (WeightStatCollection, ActStatCollection)):
                continue
            # data_in_0, data_in_1, data_in_2, ..., and (weight, bias of nn.functional)
            set_meta_arg_stat(node, entry, ActStatCollection(act_stats))
    return graph


def graph_iterator_register_stat_collections_by_type(
    graph,
    target_weight_nodes: list[str],
    target_act_nodes: list[str],
    weight_stats: dict[str, dict],
    act_stats: dict[str, dict],
    profile_output_act: bool = False,
):
    # weight stats
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in target_weight_nodes:
            continue
        if node.op != "call_module":
            logger.warning(
                f"Node {node.name} is not a call_module node, but is in target_weight_nodes. Skip."
            )
            continue
        # only create registered param/buffer stats for nn.Module
        for entry, s_meta in node.meta["mase"].parameters["software"]["args"].items():
            stat = s_meta["stat"]
            if "data_in" in entry:
                continue
            if isinstance(stat, (WeightStatCollection,)):
                continue
            set_meta_arg_stat(node, entry, WeightStatCollection(weight_stats))

    # act stats
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in target_act_nodes:
            continue
        for entry, s_meta in node.meta["mase"].parameters["software"]["args"].items():
            stat = s_meta["stat"]
            if isinstance(stat, (WeightStatCollection, ActStatCollection)):
                continue
            set_meta_arg_stat(node, entry, ActStatCollection(act_stats))
    return graph


def graph_iterator_register_stat_collections(
    graph,
    by,
    target_weight_nodes,
    target_act_nodes,
    weight_stats,
    act_stats,
    profile_output_act=False,
):
    match by:
        case "name":
            graph = graph_iterator_register_stat_collections_by_name(
                graph,
                target_weight_nodes,
                target_act_nodes,
                weight_stats,
                act_stats,
            )
        case "type":
            graph = graph_iterator_register_stat_collections_by_type(
                graph,
                target_weight_nodes,
                target_act_nodes,
                weight_stats,
                act_stats,
            )
        case _:
            raise ValueError(f"Unknown by: {by}")

    return graph


class ActProfiler(Interpreter):
    def __init__(self, module: GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

    def run_node(self, n: Node) -> Any:
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            act_stats = []
            for arg_entry in (
                n.meta["mase"].parameters["software"].get("args", {}).keys()
            ):
                stat = get_meta_arg_stat(n, arg_entry)
                if isinstance(stat, ActStatCollection):
                    act_stats.append(stat)
            if len(act_stats) > 0:
                numeric_args = tuple(
                    filter(
                        lambda x: isinstance(x, (torch.Tensor, int, float))
                        and not isinstance(x, bool),
                        args + tuple(kwargs.values()),
                    )
                )
                try:
                    assert len(numeric_args) == len(act_stats), (
                        f"Number of tensor args ({len(numeric_args)}) "
                        f"does not match number of act entries ({len(act_stats)})"
                    )
                except AssertionError:
                    breakpoint()

                for tensor_arg, stat in zip(numeric_args, act_stats):
                    stat.update(tensor_arg)

            output = getattr(self, n.op)(n.target, args, kwargs)

            # if isinstance(n.meta, _ActStatMeta):
            #     n.meta.update(output)
            return output


def graph_iterator_profile_act(graph, input_generator, num_samples):
    act_profiler = ActProfiler(graph.model, garbage_collect_values=True)

    max_batches = math.ceil(num_samples / input_generator.batch_size)

    for i in tqdm(range(max_batches), desc="Profiling act statistics"):
        batch = next(input_generator)
        act_profiler.run(*batch.values())

    return graph


def graph_iterator_profile_weight(graph):
    for node in tqdm(
        graph.fx_graph.nodes,
        total=len(list(graph.fx_graph.nodes)),
        desc="Profiling weight statistics",
    ):
        if node.op != "call_module":
            continue

        param_dict = dict(node.meta["mase"].module.named_parameters())
        buffer_dict = dict(node.meta["mase"].module.named_buffers())
        p_b_dict = {**param_dict, **buffer_dict}

        for w_name, s_meta in node.meta["mase"].parameters["software"]["args"].items():
            stat = s_meta["stat"]
            if not isinstance(stat, WeightStatCollection):
                continue

            w = p_b_dict[w_name]
            stat.update(w.data)

    return graph


def graph_iterator_compute_and_unregister_stats(graph):
    for node in graph.fx_graph.nodes:
        for entry, s_meta in (
            node.meta["mase"].parameters["software"].get("args", {}).items()
        ):
            stat = s_meta["stat"]
            if isinstance(stat, (WeightStatCollection, ActStatCollection)):
                result = stat.compute()
                set_meta_arg_stat(node, entry, result)
        # for entry, s_meta in (
        #     node.meta["mase"].parameters["software"]["results"].items()
        # ):
        #     stat = s_meta["stat"]
        #     if isinstance(stat, ActStatCollection):
        #         result = stat.compute()
        #         set_meta_result_stat(node, entry, result)
    return graph


def profile_statistics_analysis_pass(graph, pass_args: dict):
    """
    Perform profile statistics analysis on the given graph.

    :param graph: The graph to perform analysis on.
    :type graph: MaseGraph

    :param pass_args: The arguments for the analysis pass.
    :type pass_args: dict

    :return: The modified graph and an empty dictionary.
    :rtype: tuple(MaseGraph, dict)
    """

    graph = graph_iterator_register_stat_collections(
        graph,
        by=pass_args["by"],
        target_weight_nodes=pass_args["target_weight_nodes"],
        target_act_nodes=pass_args["target_activation_nodes"],
        weight_stats=pass_args["weight_statistics"],
        act_stats=pass_args["activation_statistics"],
        profile_output_act=pass_args.get("profile_output_activation", False),
    )

    graph = graph_iterator_profile_weight(graph)

    graph = graph_iterator_profile_act(
        graph,
        input_generator=pass_args["input_generator"],
        num_samples=pass_args["num_samples"],
    )

    graph = graph_iterator_compute_and_unregister_stats(graph)

    return graph, {}
