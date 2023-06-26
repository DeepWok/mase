import os
from logging import getLogger
from typing import Dict, List, Union

import regex as re
import toml
import torch
from torch.fx import Graph, GraphModule, Interpreter, Node
from tqdm import tqdm

from ...graph.mase_graph import mase_symbolic_trace
from ...utils import get_module_by_target
from ...modify.modifier import is_modifiable
from ..utils import InputArgsGenerator
from .stat import _StatBase, new_stat

logger = getLogger(__name__)


class WeightStatMeta:
    def __init__(self, name: str, real_target, stat_config: Dict[str, Dict]) -> None:
        self.name = name
        self.real_target = real_target
        if isinstance(real_target, torch.nn.Module):
            self.node_op = "call_module"
        elif isinstance(real_target, str):
            self.node_op = "call_method"
        else:
            self.node_op = "call_function"
        if "record" in stat_config:
            logger.warning(
                f"`record` (concat to record tensor) is enabled in weight profiler, which wil consumes lots of memory if the model is large"
            )
        self.stats = []
        for stat_name, stat_config in stat_config.items():
            self.stats.append(new_stat(stat_name=stat_name, **stat_config))

    def update(self, weight: torch.Tensor):
        assert isinstance(weight, torch.Tensor)
        weight = self._preprocess_weight(weight)
        for stat in self.stats:
            sample_i_tmp = weight.clone()
            stat: _StatBase
            stat.update_a_sample(sample_i_tmp)

    def _preprocess_weight(self, weight: torch.Tensor):
        return weight

    def export_to_list(self):
        results = {"op": self.node_op}
        for stat in self.stats:
            stat: _StatBase
            results |= stat.export_to_list()
        return {self.name: results}

    def export(self):
        results = {"op": self.node_op}
        for stat in self.stats:
            stat: _StatBase
            results |= stat.export()
        return {self.name: results}

    def __str__(self) -> str:
        return "WeightStatMeta(name={}, real_target={}, stats={})".format(
            self.name, self.real_target, self.stats
        )


class WeightProfiler:
    def __init__(self, module: GraphModule, config: Dict) -> None:
        self.module = module
        self.stat_configs = {}
        self.tgt_nodes = []
        self.tgt_node_patterns = []
        self.config = None
        self.no_weight_to_profile = False
        self._parse_config(config)
        self._init_meta_data()

    def _parse_config(self, config: Dict):
        assert isinstance(config, dict)
        if "weight_stats" not in config:
            self.no_weight_to_profile = True
            return
        assert isinstance(config["weight_stats"], dict)
        self.stat_configs = config["weight_stats"]

        if "weight_nodes" in config:
            self.tgt_nodes = config["weight_nodes"]
        if "weight_node_patterns" in config:
            self.tgt_node_patterns = config["weight_node_patterns"]

        if len(self.stat_configs) == 0 or (
            len(self.tgt_nodes) + len(self.tgt_node_patterns) == 0
        ):
            self.no_weight_to_profile = True
        else:
            logger.info(
                "weight statistics to profile: {}".format(
                    list(self.stat_configs.keys())
                )
            )

        self.config = config

    def _match_a_node_name_pattern(self, node_name: str):
        pattern_is_matched = False
        for pattern in self.tgt_node_patterns:
            match = re.fullmatch(pattern, node_name)
            pattern_is_matched = bool(match)
            if pattern_is_matched:
                break
        return pattern_is_matched

    def _init_meta_data(self):
        if self.no_weight_to_profile:
            return
        for node in self.module.graph.nodes:
            node.meta = {}
            if node.op != "call_module":
                continue

            if not (
                node.target in self.tgt_nodes
                or self._match_a_node_name_pattern(node.target)
            ):
                continue

            real_target = get_module_by_target(self.module, node.target)
            for name, weight in real_target.named_parameters():
                node.meta[name] = WeightStatMeta(
                    name=node.target + "::" + name,
                    real_target=real_target,
                    stat_config=self.stat_configs,
                )

    def profile(self):
        if self.no_weight_to_profile:
            return
        for node in tqdm(
            self.module.graph.nodes, total=len(list(self.module.graph.nodes))
        ):
            if node.op != "call_module":
                continue

            if not (
                node.target in self.tgt_nodes
                or self._match_a_node_name_pattern(node.target)
            ):
                continue

            real_target = get_module_by_target(self.module, node.target)

            for name, weight in real_target.named_parameters():
                node.meta[name].update(weight)

    def export_profile(self, save_path: str = None):
        if self.no_weight_to_profile:
            logger.warning("No weight to profile, skip export profile")
            return {}

        stat_dict = {}
        for node in self.module.graph.nodes:
            if len(node.meta) > 0:
                for k, v in node.meta.items():
                    if isinstance(v, WeightStatMeta):
                        stat_dict |= v.export()

        if save_path is not None:
            with open(save_path, "w") as f:
                toml.dump(stat_dict, f)
        return stat_dict

    def export_profile_to_list(self, save_path: str = None):
        if self.no_weight_to_profile:
            logger.warning("No weight to profile, skip export profile")
            return {}

        stat_dict = {}
        for node in self.module.graph.nodes:
            if len(node.meta) > 0:
                for k, v in node.meta.items():
                    if isinstance(v, WeightStatMeta):
                        stat_dict |= v.export_to_list()

        if save_path is not None:
            with open(save_path, "w") as f:
                toml.dump(stat_dict, f)
        return stat_dict

    def create_config_template(self, modifiable_only=True, save_path: str = None):
        template = {
            "weight_nodes": [],
            "weight_node_patterns": [],
            "weight_stats": {
                "reduced_soft_range": {"num_sigmas": 3},
            },
        }
        for node in self.module.graph.nodes:
            if modifiable_only:
                if not is_modifiable(node, self.module):
                    continue
            if node.op != "call_module":
                continue
            real_target = get_module_by_target(self.module, node.target)

            if len(dict(real_target.named_parameters())) == 0:
                continue
            template["weight_nodes"].append(node.target)
        if save_path is not None:
            with open(save_path, "w") as f:
                toml.dump(template, f)
        return template


def run_weight_profiler(
    model: torch.nn.Module,
    dummy_inputs_for_fx,
    config_path: str,
    save_dir: str,
):
    is_training = model.training
    gm = mase_symbolic_trace(model, dummy_inputs_for_fx)

    assert os.path.isfile(config_path) and config_path.endswith(".toml")
    config = toml.load(config_path)
    profiler = WeightProfiler(gm, config)

    config_template_save_path = os.path.join(save_dir, "profile_weight_template.toml")
    profiler.create_config_template(
        modifiable_only=True, save_path=config_template_save_path
    )
    logger.info(
        f"Weight profiler config template is saved to {config_template_save_path}"
    )

    if profiler.no_weight_to_profile:
        logger.info("No weight to profile, skip weight profiling")
        return {}

    gm.eval()
    logger.info("Traversing model to collect weight statistics...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gm.to(device)
    with torch.no_grad():
        profiler.profile()

    if is_training:
        gm.train()
    profile_save_path = os.path.join(save_dir, "weight_profile.toml")

    weight_profile = profiler.export_profile_to_list(save_path=profile_save_path)
    logger.info(f"Weight profile is saved to {profile_save_path}")
    return weight_profile
