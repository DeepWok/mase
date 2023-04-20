import os
from logging import getLogger
from typing import List, Union

import regex as re
import toml
import torch
from torch.fx import Graph, GraphModule, Interpreter, Node
from tqdm import tqdm

from ...graph.mase_graph import mase_symbolic_trace
from ..utils import InputArgsGenerator
from .stat import STAT_NAME_TO_CLS, _StatBase

logger = getLogger(__name__)


class _StatNodeMeta:
    def __init__(self, node_name: str, stats: List[str]) -> None:
        self.name = node_name
        if "record" in stats:
            logger.warning(
                f"`record` (concat to record tensor) is enabled in activation profiler, which wil consumes lots of memory as the num of batch increases"
            )
        self.stats = []
        for stat in stats:
            assert stat in STAT_NAME_TO_CLS
            self.stats.append(STAT_NAME_TO_CLS[stat]())

    def update(self, batch: torch.Tensor):
        assert isinstance(batch, torch.Tensor)

        batch = torch.flatten(batch, start_dim=0, end_dim=-2)  # 2D x
        for i in range(batch.shape[0]):
            sample_i = batch[[i], ...]
            for stat in self.stats:
                stat: _StatBase
                stat.update_a_sample(sample_i)

    def export_to_list(self):
        results = {}
        for stat in self.stats:
            stat: _StatBase
            results |= stat.export_to_list()
        return {self.name: results}

    def export(self):
        results = {}
        for stat in self.stats:
            stat: _StatBase
            results |= stat.export()
        return {self.name: results}


class ActivationProfiler(Interpreter):
    def __init__(
        self,
        graph_module: GraphModule,
        config: dict,
        garbage_collect_values: bool = True,
    ):
        assert isinstance(graph_module, GraphModule)
        super().__init__(graph_module, garbage_collect_values)
        self.stat_names = []
        self.tgt_nodes = []
        self.tgt_node_patterns = []
        self.num_profiled_batches = 0
        self.config = None
        self._parse_config(config)
        self._init_meta_data()

    def _parse_config(self, config: dict):
        assert isinstance(config, dict)
        assert "act_stats" in config
        assert isinstance(config["act_stats"], (tuple, list))
        self.stat_names = list(config["act_stats"])
        if "act_nodes" in config:
            self.tgt_nodes = config["act_nodes"]
        if "act_node_patterns" in config:
            self.tgt_node_patterns = config["act_node_patterns"]
        assert len(self.stat_names) > 0
        assert (
            len(self.tgt_node_patterns) + len(self.tgt_nodes) > 0
        ), f"No available `act_nodes` or `act_node_patterns` in {config}"
        self.config = config

    def _match_a_node_name_pattern(self, node_name):
        pattern_is_matched = False
        for pattern in self.tgt_node_patterns:
            match = re.fullmatch(pattern, node_name)
            pattern_is_matched = bool(match)
            if pattern_is_matched:
                break
        return pattern_is_matched

    def _init_meta_data(self):
        for node in self.module.graph.nodes:
            if node.op in ["call_function", "call_method"]:
                if node.name in self.tgt_nodes or self._match_a_node_name_pattern(
                    node.name
                ):
                    node.meta = _StatNodeMeta(node.name, self.stat_names)
            elif node.op == "call_module":
                if node.target in self.tgt_nodes or self._match_a_node_name_pattern(
                    node.target
                ):
                    node.meta = _StatNodeMeta(node.target, self.stat_names)

    def run_node(self, n: Node):
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            output = getattr(self, n.op)(n.target, args, kwargs)

            if isinstance(n.meta, _StatNodeMeta):
                n.meta.update(output)
            return output

    def profile_a_batch(self, *args):
        outputs = self.run(*args)
        self.num_profiled_batches += 1
        return outputs

    def export_profile_to_list(self, save_path: str = None):
        stat_dict = {}
        for node in self.module.graph.nodes:
            if isinstance(node.meta, _StatNodeMeta):
                stat_dict |= node.meta.export_to_list()
        if save_path is not None:
            with open(save_path, "w+") as f:
                toml.dump(stat_dict, f)
        return stat_dict

    def export_profile(self, save_path: str = None):
        stat_dict = {}
        for node in self.module.graph.nodes:
            if isinstance(node.meta, _StatNodeMeta):
                stat_dict |= node.meta.export()
        if save_path is not None:
            with open(save_path, "w+") as f:
                toml.dump(stat_dict, f)
        return stat_dict

    def create_config_template(self, save_path: str = None):
        template = {
            "act_nodes": [],
            "act_num_batches_to_profile": 4,
            "act_stats": ["variance"],
        }
        for node in self.module.graph.nodes:
            if node.op in ["call_function", "call_method"]:
                template["act_nodes"].append(node.name)
            elif node.op == "call_module":
                template["act_nodes"].append(node.target)
        if save_path is not None:
            with open(save_path, "w+") as f:
                toml.dump(template, f)
        return template


def run_activation_profiler(
    model_name,
    task,
    model: torch.nn.Module,
    data_module,
    dummy_inputs_for_fx,
    config_path: str,
    save_dir,
):
    is_training = model.training
    input_generator = InputArgsGenerator(
        model_name=model_name, task=task, data_module=data_module
    )
    gm = mase_symbolic_trace(root=model, concrete_args=dummy_inputs_for_fx)
    assert os.path.isfile(config_path) and config_path.endswith(".toml")

    config = toml.load(config_path)
    profiler = ActivationProfiler(graph_module=gm, config=config)

    config_template_save_path = os.path.join(save_dir, "profile_act_template.toml")
    profiler.create_config_template(save_path=config_template_save_path)
    logger.info(
        f"activation profiler config template is saved at {config_template_save_path}"
    )

    num_batches_to_profile = config["act_num_batches_to_profile"]

    gm.eval()
    logger.info("Feeding inputs to model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gm.to(device)
    with torch.no_grad():
        for i in tqdm(range(num_batches_to_profile)):
            input_args = next(input_generator)
            input_args = [arg.to(device) for arg in input_args]
            _ = profiler.profile_a_batch(*input_args)
    if is_training:
        gm.train()
    profile_save_path = os.path.join(save_dir, "act_profile.toml")
    profiler.export_profile_to_list(save_path=profile_save_path)
    logger.info(f"activation profile is saved at {profile_save_path}")
