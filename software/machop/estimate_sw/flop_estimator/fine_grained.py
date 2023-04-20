import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import toml
import torch
from torch.fx import Graph, GraphModule, Interpreter
from torch.fx.node import Argument, Target

from ...graph.fx_profiler import GraphProfiler
from ...graph.mase_tracer import MaseTracer
from ..utils import get_input_args
from .calculator import calculate_funcs, calculate_modules

logger = logging.getLogger(__name__)


def estimate_sw_fine_grained(
    model_name: int,
    task: str,
    info: dict,
    model: torch.nn.Module,
    data_module,
    config: dict = None,
    save_dir: str = None,
):
    assert isinstance(
        config["ignore_modules"], (list, tuple, set)
    ), "ignore_modules should be a list/tuple/set of string nn.Module names"
    logger.debug(f"estimate-sw config: {config}")

    _, input_args = get_input_args(model_name, model, task, data_module, info)

    output_file = config["output_file"]
    graph: Graph = MaseTracer().trace(model, input_args)
    graph_module = GraphModule(model, graph)
    profiler = GraphProfiler(graph_module)
    profiler.set_function_wrapper(calculate_funcs)
    profiler.set_module_wrapper(calculate_modules)

    _ = profiler.run(*input_args)

    data = profiler.stats.data
    logger.info("Estimate-sw result")
    print(data)

    output_file = os.path.join(save_dir, output_file)
    with open(output_file, "w") as toml_file:
        toml.dump(data, toml_file)
    logger.info(f"Estimate-sw report saved to {output_file}")
