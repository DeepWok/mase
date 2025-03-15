from torch.fx import Interpreter
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from typing import Any
import torch
import torch.nn.functional as F
from machop.chop.tools.get_input import InputGenerator
from .utils import get_quant_cfg
import json

QUANTIZEABLE_OP = (
    # "add",
    # "bmm",
    # "conv1d",
    # "conv2d",
    "matmul",
    # "mul",
    "linear",
    # "relu",
    # "sub",
)

SEARCH_TASK = (
    "quantile_int",
    "quantile_scale_int",
    "bfp",
    "softmax",
)

from machop.chop.passes.graph.transforms import (
    quantize_transform_pass,
)
from machop.chop.passes.graph.utils import get_mase_op
from .SearchSpace import SearchSpaceBase, QuantileSearchSpace, SoftmaxSearchSpace
from .SearchStrategy import SearchStrategyBase, SoftmaxSearch
import copy
from tqdm import tqdm


class LayerQuantProfiler(Interpreter):
    def __init__(
        self,
        module: GraphModule,
        garbage_collect_values: bool = True,
        search_info: str = None,
    ):
        super().__init__(module, garbage_collect_values)
        self.search_info = search_info

    def run_node(self, n: Node) -> Any:
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            output = getattr(self, n.op)(n.target, args, kwargs)

            task = self.search_info["task"]
            assert task in SEARCH_TASK, f"{task} is not supported right now"

            config_choice = self.search_info["config_choice"]
            config_list = self.search_info["config_list"]
            match task:
                case "quantile_scale_int":
                    if get_mase_op(n) in QUANTIZEABLE_OP:
                        space = QuantileSearchSpace(
                            config_choice, config_list, "scale_integer"
                        )
                        search = SearchStrategyBase(n, args, kwargs, space)
                        output = search(output, 2)
                case "quantile_int":
                    if get_mase_op(n) in QUANTIZEABLE_OP:
                        space = QuantileSearchSpace(
                            config_choice, config_list, "integer"
                        )
                        search = SearchStrategyBase(n, args, kwargs, space)
                        output = search(output, 2)
                case "bfp":
                    if get_mase_op(n) in QUANTIZEABLE_OP:
                        space = SearchSpaceBase(config_choice, config_list, "block_fp")
                        search = SearchStrategyBase(n, args, kwargs, space)
                        output = search(output, 2)
                case "softmax":
                    if get_mase_op(n) in ("softmax", "hash_softmax"):
                        space = SoftmaxSearchSpace(
                            config_choice, config_list, "integer"
                        )
                        search_spaces = space.build_search_space(
                            get_mase_op(n)
                        )
                        search = SoftmaxSearch(n, args, kwargs, search_spaces)
                        output = search(output, 2)
            return output


def _in_layer_quant_search(mg, info, search_info):
    act_profiler = LayerQuantProfiler(
        mg.model, garbage_collect_values=True, search_info=search_info
    )

    input_generator = InputGenerator(
        model_info=info["model_info"],
        data_module=info["data_module"],
        task="cls",
        which_dataloader="train",
    )

    batch = next(input_generator)
    act_profiler.run(*batch.values())

    quant_cfg = get_quant_cfg(mg)
    return quant_cfg
