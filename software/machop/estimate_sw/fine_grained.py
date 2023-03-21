import torch
import logging
import os

import toml

from torch.fx import Graph, GraphModule
from torch.fx import Interpreter
from torch.fx.node import Argument, Target

from .utils import get_input_args
from ..graph.mase_tracer import MaseTracer
from .calculator import calculate_funcs, calculate_modules

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from collections import defaultdict


class Stats:
    data = defaultdict(dict)

    def add_data(self, key, value):
        self.data[key] = value


class GraphProfiler(Interpreter):
    stats = Stats()

    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_function`` node and return the result.
        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        # Execute the function and return the result

        out_data = target(*args, **kwargs)
        meta = calculate_funcs(target, args, kwargs, out_data)
        self.stats.add_data(target.__name__, meta)
        return out_data

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_module`` node and return the result.
        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        in_data = args[0]
        # if isinstance(submod, torch.nn.ReLU):
        #     import pdb

        #     pdb.set_trace()
        out_data = submod(*args, **kwargs)
        meta = calculate_modules(submod, in_data, out_data)
        self.stats.add_data(target, meta)
        return out_data
        # return submod(*args, **kwargs)


def estimate_sw_fine_grained(
    model_name: int,
    info: dict,
    model: torch.nn.Module,
    task: str,
    data_loader,
    save_path: str = None,
    config: dict = None,
):
    assert isinstance(
        config["ignore_modules"], (list, tuple, set)
    ), "ignore_modules should be a list/tuple/set of string nn.Module names"
    logging.debug(f"estimate-sw config: {config}")

    _, input_args = get_input_args(model_name, model, task, data_loader, info)

    output_file = config["output_file"]
    graph: Graph = MaseTracer().trace(model, input_args)
    graph_module = GraphModule(model, graph)
    profiler = GraphProfiler(graph_module)

    _ = profiler.run(*input_args)

    data = profiler.stats.data

    output_file = os.path.join(save_path, output_file)
    with open(output_file, "w") as toml_file:
        toml.dump(data, toml_file)
