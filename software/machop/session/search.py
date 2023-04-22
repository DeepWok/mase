import copy
import logging
import os
import random

import toml
import torch
from hyperopt import fmin, hp, rand, tpe
from machop.graph.mase_graph import _get_prev_call_node, get_module_by_name
from machop.models import nlp_models
from machop.modify.modifier import Modifier, is_modifiable
from machop.modify.quantizers.quantizers import integer_fraction
from machop.utils import to_numpy
from torchvision.models.feature_extraction import (
    NodePathTracer,
    create_feature_extractor,
    get_graph_node_names,
)
from tqdm import tqdm

from .plt_wrapper import get_model_wrapper

logger = logging.getLogger(__name__)


class SearchBase:
    def __init__(
        self,
        model_name,
        info,
        model,
        task,
        modifier_kwargs,
        data_module,
        search_config,
        save_dir,
        accelerator="auto",
    ) -> None:
        self.model_name = model_name
        self.info = info
        self.model = model
        self.task = task
        self.modifier_kwargs = modifier_kwargs
        self._parse_config(search_config)
        self._prepare_loader(data_module)
        self.modifier = Modifier(
            model=self.model["model"] if self._is_nlp_model(model_name) else self.model,
            config_path=None,
            silent=True,
            **modifier_kwargs,
        )
        self.save_dir = save_dir
        if accelerator == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif accelerator == "gpu":
            self.device = torch.device("cuda:0")
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise RuntimeError(f"Unsupported accelerator {accelerator}")

    def _parse_config(self, search_config):
        with open(search_config, "r") as f:
            search_args = toml.load(f)
        # building search space
        self.search_strategy = search_args["strategy"]["name"]
        self.search_data = search_args["search_data"]
        self.search_strategy_config = search_args["strategy_config"]
        self.search_space = search_args["search_space"]
        self.which_dataloader = search_args["search_data"].get(
            "dataloader", "train_dataloader"
        )
        assert self.which_dataloader in [
            "train_dataloader",
            "val_dataloader",
            "test_dataloader",
        ]

    def _is_nlp_model(self, model_name) -> bool:
        return model_name in nlp_models

    def _prepare_loader(self, data_module):
        data_module.setup()
        data_module.prepare_data()

        self.data_loader = getattr(data_module, self.which_dataloader)()
        # self.data_loader = data_module.train_dataloader()
        self.num_batches = self.search_data["num_batches"]

    def rebuild_modifier(self):
        self.modifier = Modifier(
            model=self.model["model"]
            if self._is_nlp_model(self.model_name)
            else self.model,
            config_path=None,
            silent=True,
            **self.modifier_kwargs,
        )

    def search(self):
        raise NotImplementedError


class SearchQuantization(SearchBase):
    default_config = {
        "bias_frac_width": 5,
        "bias_width": 8,
        "data_in_frac_width": 5,
        "data_in_width": 8,
        "name": "integer",
        "weight_frac_width": 3,
        "weight_width": 8,
    }

    def build_search_space(self):
        function_nodes_to_modify = {}
        module_nodes_to_modify = {}
        method_nodes_to_modify = {}

        search_space = self.search_space

        self.modifier.build_graph()
        self.modifier.check_modifiable()
        for n in self.modifier.graph.nodes:
            if n.op == "call_function":
                if is_modifiable(
                    n,
                    self.model["model"]
                    if self._is_nlp_model(self.model_name)
                    else self.model,
                ):
                    function_nodes_to_modify[n.name] = dict(search_space)
                else:
                    continue
            elif n.op == "call_module":
                if is_modifiable(
                    n,
                    self.model["model"]
                    if self._is_nlp_model(self.model_name)
                    else self.model,
                ):
                    module_nodes_to_modify[n.target] = dict(search_space)
                else:
                    continue
            elif n.op == "call_method":
                if is_modifiable(
                    n,
                    self.model["model"]
                    if self._is_nlp_model(self.model_name)
                    else self.model,
                ):
                    method_nodes_to_modify[n.name] = dict(search_space)
                else:
                    continue
        self.full_search_space = {
            "function_nodes_to_modify": function_nodes_to_modify,
            "module_nodes_to_modify": module_nodes_to_modify,
            "method_nodes_to_modify": method_nodes_to_modify,
        }

    # this is maybe useful if one wants to try different search strategies
    def random_sample_search_space(self):
        sampled_search_space = {}
        for meta_name, meta_value in self.full_search_space.items():
            sampled_search_space[meta_name] = {}
            for node, search_space in meta_value.items():
                sampled_search_space[meta_name][node] = {}
                for key, value in search_space.items():
                    if isinstance(value, list):
                        picked = random.choice(value)
                    elif isinstance(value, dict):
                        picked = random.choice(list(value.values()))
                    else:
                        raise NotImplementedError
                    sampled_search_space[meta_name][node][key] = picked
        return sampled_search_space

    def get_config_instance_and_evaluate(self, sampled_search_space):
        self.modifier.load_config(None)
        sampled_search_space["default"] = self.default_config
        self.modifier.load_config(sampled_search_space)
        graph_module = self.modifier.modify()

        if self._is_nlp_model(self.model_name):
            model = copy.deepcopy(self.model)
            model.update(model=graph_module)
        else:
            model = graph_module

        wrapper_cls = get_model_wrapper(name=self.model_name, task=self.task)
        pl_model = wrapper_cls(model, info=self.info).to(self.device)
        losses = []
        for batch_idx, batch in enumerate(self.data_loader):
            if batch_idx >= self.num_batches:
                break

            batch = self._move_batch_to_device(batch)
            outputs = pl_model.training_step(batch, batch_idx)
            losses.append(torch.mean(outputs["loss"]).item())
        return sum(losses) / len(losses)

    def _move_batch_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            return [
                x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
            ]
        else:
            raise RuntimeError

    def rebuild_hyper_opt_search_space(self):
        sampled_search_space = {}
        for meta_name, meta_value in self.full_search_space.items():
            sampled_search_space[meta_name] = {}
            for node, search_space in meta_value.items():
                sampled_search_space[meta_name][node] = {}
                for key, value in search_space.items():
                    if isinstance(value, list):
                        picked = hp.choice(
                            meta_name + "/" + node + "/" + key, list(value)
                        )
                    else:
                        raise NotImplementedError
                    sampled_search_space[meta_name][node][key] = picked
        return sampled_search_space

    def convert_best_to_config(self, best):
        config = {
            "default": self.default_config,
            "function_nodes_to_modify": {},
            "module_nodes_to_modify": {},
            "method_nodes_to_modify": {},
        }

        for entry, value in best.items():
            meta_name, node, key = entry.split("/")
            if node not in config[meta_name]:
                if key == "name":
                    config[meta_name][node] = {key: "integer"}
                else:
                    config[meta_name][node] = {key: int(value)}
            else:
                if key == "name":
                    config[meta_name][node][key] = "integer"
                else:
                    config[meta_name][node][key] = int(value)
        return config

    def hyper_opt_strategy(self):
        space = self.rebuild_hyper_opt_search_space()

        def objective(my_instance):
            self.rebuild_modifier()
            loss = self.get_config_instance_and_evaluate(my_instance)
            return loss

        if self.search_strategy_config["algorithm"] == "tpe":
            algo = tpe.suggest
        elif self.search_strategy_config["algorithm"] == "random":
            algo = rand.suggest
        else:
            raise NotImplementedError

        max_evals = self.search_strategy_config["max_evals"]

        best = fmin(objective, space=space, algo=algo, max_evals=max_evals)
        best = self.convert_best_to_config(best)
        return best

    def analytical_quantize(self):
        """
        Get frac width from non-clipping width given a constant bitwidth
        """
        self.model.eval()
        nodes = get_graph_node_names(self.model)[1]
        graph_module = create_feature_extractor(self.model, nodes)
        tracer = NodePathTracer()
        tracer.trace(self.model)
        name_mapping = {v: str(k) for k, v in tracer.node_to_qualname.items()}

        # gather dynamic range
        out_stats = {}
        graph_module.to(self.device)
        for i, data in tqdm(enumerate(self.data_loader), total=self.num_batches):
            # for i, data in enumerate(self.data_loader):
            input_data, label = data
            input_data, label = self._move_batch_to_device([input_data, label])
            if i >= self.num_batches:
                break
            outputs = graph_module(input_data)
            for node_name, out_feature in outputs.items():
                node_name = name_mapping[node_name]
                min_value = torch.min(out_feature).item()
                max_value = torch.max(out_feature).item()
                if node_name not in out_stats.keys():
                    out_stats[node_name] = {"min": min_value, "max": max_value}
                else:
                    out_stats[node_name]["min"] = min(
                        out_stats[node_name]["min"], min_value
                    )

                    out_stats[node_name]["max"] = max(
                        out_stats[node_name]["max"], max_value
                    )

        # check all producers
        in_stats = {}
        for n in graph_module.graph.nodes:
            for prev_node in _get_prev_call_node(n, []):
                assert prev_node.name in out_stats.keys(), f"{prev_node.name} not found"
                if n not in in_stats.keys():
                    in_stats[n] = out_stats[prev_node.name]
                else:
                    in_stats[n]["min"] = min(
                        in_stats[n]["min"], out_stats[prev_node.name]["min"]
                    )
                    in_stats[n]["max"] = max(
                        in_stats[n]["max"], out_stats[prev_node.name]["max"]
                    )

        # analytical decision on fraction width
        config = {
            "default": self.default_config,
            "module_nodes_to_modify": {},
            "function_nodes_to_modify": {},
            "method_nodes_to_modify": {},
        }
        assert (
            len(self.search_space["data_in_width"]) == 1
            and self.search_space["data_in_width"][0]
            == self.default_config["data_in_width"]
        )
        assert (
            len(self.search_space["weight_width"]) == 1
            and self.search_space["weight_width"][0]
            == self.default_config["weight_width"]
        )
        assert (
            len(self.search_space["bias_width"]) == 1
            and self.search_space["bias_width"][0] == self.default_config["bias_width"]
        )

        for n in graph_module.graph.nodes:
            if not is_modifiable(node=n, model=self.model):
                # not all nodes support sw-modification
                continue
            if n.op in ["call_function", "call_module", "call_method"]:
                node_config = copy.deepcopy(self.default_config)
                assert (
                    node_config["name"] == "integer"
                ), "currently analytical search only supports fixed point quantization"
                data_min = in_stats[n]["min"]
                data_max = in_stats[n]["max"]
                node_config["data_in_frac_width"] = integer_fraction(
                    self.search_space["data_in_width"][0],
                    self.search_space["data_in_frac_width"],
                    data_min,
                    data_max,
                )
                m = get_module_by_name(self.model, n.target)
                if m:
                    for param_name, parameter in m.named_parameters():
                        if "weight" in param_name:
                            weight_min = torch.min(parameter).item()
                            weight_max = torch.max(parameter).item()
                            node_config["weight_frac_width"] = integer_fraction(
                                self.search_space["weight_width"][0],
                                self.search_space["weight_frac_width"],
                                weight_min,
                                weight_max,
                            )
                        if "bias" in param_name:
                            bias_min = torch.min(parameter).item()
                            bias_max = torch.max(parameter).item()
                            node_config["bias_frac_width"] = integer_fraction(
                                self.search_space["bias_width"][0],
                                self.search_space["bias_frac_width"],
                                bias_min,
                                bias_max,
                            )
                if n.op == "call_function":
                    config["function_nodes_to_modify"][n.name] = node_config
                elif n.op == "call_method":
                    config["method_nodes_to_modify"][n.name] = node_config
                else:
                    config["module_nodes_to_modify"][n.target] = node_config

        return config

    def strategies(self):
        if self.search_strategy == "hyperopt":
            return self.hyper_opt_strategy()
        elif self.search_strategy == "analytical":
            return self.analytical_quantize()
        else:
            raise NotImplementedError

    def search(self):
        if self.search_strategy != "analytical":
            self.build_search_space()
        best = self.strategies()
        output_file = self.search_strategy_config["output_file"]
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            output_file = os.path.join(self.save_dir, output_file)
            with open(output_file, "w") as f:
                toml.dump(best, f)
            logger.info("Search result saved to {}".format(output_file))


def search(
    model_name,
    info,
    model,
    task,
    modifier_kwargs,
    data_module,
    search_config,
    save_dir,
    accelerator,
):
    # model_name, info, model, task, data_module, search_config, save_dir
    logger.info("Search started ... ")
    searcher = SearchQuantization(
        model_name=model_name,
        info=info,
        model=model,
        task=task,
        modifier_kwargs=modifier_kwargs,
        data_module=data_module,
        search_config=search_config,
        save_dir=save_dir,
        accelerator=accelerator,
    )
    searcher.search()
