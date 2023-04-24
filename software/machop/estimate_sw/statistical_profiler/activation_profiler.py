import os
from logging import getLogger
from typing import Dict, List, Union

import regex as re
import toml
import torch
from torch.fx import Graph, GraphModule, Interpreter, Node
from tqdm import tqdm

from ...graph.mase_graph import mase_symbolic_trace
from ...modify.modifier import is_modifiable
from ..utils import InputArgsGenerator
from .stat import _StatBase, new_stat

logger = getLogger(__name__)


class ActStatMeta:
    def __init__(self, name: str, real_target, stat_configs: Dict[str, Dict]) -> None:
        self.name = name
        self.real_target = real_target
        if "record" in stat_configs:
            logger.warning(
                f"`record` (concat to record tensor) is enabled in activation profiler, which wil consumes lots of memory as the num of batch increases"
            )
        self.stats = []
        for stat_name, stat_config in stat_configs.items():
            self.stats.append(new_stat(stat_name=stat_name, **stat_config))

    def update(self, batch: torch.Tensor):
        assert isinstance(batch, torch.Tensor)

        batch = self._preprocess_batch(batch)
        for i in range(batch.shape[0]):
            sample_i = batch[[i], ...]
            for stat in self.stats:
                sample_i_tmp = sample_i.clone()
                stat: _StatBase
                stat.update_a_sample(sample_i_tmp)

    def _preprocess_batch(self, batch: torch.Tensor):
        assert isinstance(batch, torch.Tensor)
        if isinstance(self.real_target, torch.nn.Linear):
            # 2D x
            batch = batch.flatten(start_dim=0, end_dim=-2)
        else:
            # 1D x
            batch = batch.flatten()
        return batch

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

    def __str__(self) -> str:
        return "ActStatMeta(name={}, real_target={}, stats={})".format(
            self.name, self.real_target, (str(stat) for stat in self.stats)
        )


class ActivationProfiler(Interpreter):
    def __init__(
        self,
        graph_module: GraphModule,
        config: dict,
        garbage_collect_values: bool = True,
    ):
        assert isinstance(graph_module, GraphModule)
        super().__init__(graph_module, garbage_collect_values)
        self.stat_configs = {}
        self.tgt_nodes = []
        self.tgt_node_patterns = []
        self.num_profiled_batches = 0
        self.config = None

        self.no_act_to_profile = False
        self._parse_config(config)
        self._init_meta_data()

    def _parse_config(self, config: dict):
        assert isinstance(config, dict)
        if not "act_stats" in config:
            self.no_act_to_profile = True
            return
        assert isinstance(config["act_stats"], dict)
        self.stat_configs = config["act_stats"]
        if "act_nodes" in config:
            self.tgt_nodes = config["act_nodes"]
        if "act_node_patterns" in config:
            self.tgt_node_patterns = config["act_node_patterns"]
        if len(self.stat_configs) == 0 or (
            len(self.tgt_node_patterns) + len(self.tgt_nodes) == 0
        ):
            self.no_act_to_profile = True
        else:
            logger.info(
                "activation statistics to be profiled: {}".format(
                    list(self.stat_configs.keys())
                )
            )
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
        if self.no_act_to_profile:
            return
        for node in self.module.graph.nodes:
            node.meta = {}
            if node.op not in ["call_function", "call_method", "call_module"]:
                continue

            if node.op in ["call_function", "call_method"]:
                if not (
                    node.name in self.tgt_nodes
                    or self._match_a_node_name_pattern(node.name)
                ):
                    continue
                node_name = node.name
            if node.op == "call_module":
                if not (
                    node.target in self.tgt_nodes
                    or self._match_a_node_name_pattern(node.target)
                ):
                    continue
                node_name = node.target

            if node.op in ["call_function", "call_method"]:
                real_target = node.target
            elif node.op == "call_module":
                real_target = self.fetch_attr(node.target)

            num_input_nodes = len(node.all_input_nodes)
            for i in range(num_input_nodes):
                node.meta[f"data_in_{i}"] = ActStatMeta(
                    f"{node_name}::data_in_{i}",
                    real_target=real_target,
                    stat_configs=self.stat_configs,
                )

    def run_node(self, n: Node):
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            if len(n.meta) > 0:
                tensor_args = args + tuple(kwargs.values())
                for i, tensor_arg in enumerate(tensor_args):
                    if isinstance(tensor_arg, torch.Tensor):
                        n.meta[f"data_in_{i}"].update(tensor_arg)

            output = getattr(self, n.op)(n.target, args, kwargs)

            # if isinstance(n.meta, _ActStatMeta):
            #     n.meta.update(output)
            return output

    def profile_a_batch(self, *args):
        outputs = self.run(*args)
        self.num_profiled_batches += 1
        return outputs

    def export_profile_to_list(self, save_path: str = None):
        stat_dict = {}
        for node in self.module.graph.nodes:
            if len(node.meta) > 0:
                for k, v in node.meta.items():
                    if isinstance(v, ActStatMeta):
                        stat_dict |= v.export_to_list()

        if save_path is not None:
            with open(save_path, "w+") as f:
                toml.dump(stat_dict, f)
        return stat_dict

    def export_profile(self, save_path: str = None):
        stat_dict = {}
        for node in self.module.graph.nodes:
            if len(node.meta) > 0:
                for k, v in node.meta.items():
                    if isinstance(v, ActStatMeta):
                        stat_dict |= v.export()
        if save_path is not None:
            with open(save_path, "w+") as f:
                toml.dump(stat_dict, f)
        return stat_dict

    def create_config_template(self, save_path: str = None):
        template = {
            "act_nodes": [],
            "act_num_batches_to_profile": 4,
            "act_stats": {
                "variance": {"offload_to_cpu": True},
                "soft_range": {"num_sigmas": 2},
            },
        }
        for node in self.module.graph.nodes:
            if not is_modifiable(node, self.module):
                continue
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
    if profiler.no_act_to_profile:
        logger.info("No activation to profile")
        return

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
