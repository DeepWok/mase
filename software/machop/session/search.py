import logging
import os
import random

import toml
from hyperopt import fmin, hp, rand, tpe
from machop.models import vision_models
from machop.modify.modifier import Modifier
from machop.utils import to_numpy

logger = logging.getLogger(__name__)


class SearchBase:
    def __init__(self, model_name, model, data_module, search_config, save_dir) -> None:
        self.model_name = model_name
        self.model = model
        self._parse_config(search_config)
        self._prepare_loader(data_module)
        self.modifier = Modifier(model=self.model, config_path=None, silent=True)
        self.save_dir = save_dir

    def _parse_config(self, search_config):
        with open(search_config, "r") as f:
            search_args = toml.load(f)
        # building search space
        self.search_space = search_args["search_space"]
        self.search_data = search_args["search_data"]
        self.search_strategy = search_args["strategy"]["name"]
        self.search_strategy_config = search_args["strategy_config"]

    def _is_vision_model(self, model_name) -> bool:
        return model_name in vision_models

    def _prepare_loader(self, data_module):
        if self._is_vision_model(self.model_name):
            data_module.setup()
            data_module.prepare_data()
        else:
            raise NotImplementedError(f"Model {self.model_name} is not supported yet.")
        self.data_loader = getattr(data_module, self.search_data["dataloader"])()
        self.num_batches = self.search_data["num_batches"]

    def rebuild_modifier(self):
        self.modifier = Modifier(model=self.model, config_path=None, silent=True)

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
        functions_to_modify, modules_to_modify, methods_to_modify = {}, {}, {}
        search_space = self.search_space
        for n in self.modifier.graph.nodes:
            if n.op == "call_function":
                # TODO: support node-wise function?
                if "add" in n.name:
                    tmp_name = "add"
                elif "relu" in n.name:
                    tmp_name = "relu"
                    # FIXME: tmp_name is not always assigned
                functions_to_modify[tmp_name] = dict(search_space)
            elif n.op == "call_module":
                # TODO: batchnorm is skipped in modify?
                if "bn" in n.target:
                    continue
                elif "pool" in n.target:
                    continue
                elif "downsample" in n.target:
                    continue
                modules_to_modify[n.target] = dict(search_space)
            elif n.op == "call_method":
                methods_to_modify = dict(search_space)
        self.full_search_space = {
            "functions_to_modify": functions_to_modify,
            "modules_to_modify": modules_to_modify,
            "methods_to_modify": methods_to_modify,
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
        sampled_search_space["module_classes_to_modify"] = {}
        self.modifier.load_config(sampled_search_space)
        self.modifier.check_modifiable(self.model)
        graph_module = self.modifier.modify()
        accs = []
        for i, data in enumerate(self.data_loader):
            input_data, label = data
            if i >= self.num_batches:
                break
            logits = graph_module(input_data)
            acc = sum(logits.argmax(dim=1) == label) / label.numel()
            accs.append(acc)
        # average accs
        return sum(accs) / len(accs)

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
            "module_classes_to_modify": {},
            "functions_to_modify": {},
            "modules_to_modify": {},
            "methods_to_modify": {},
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
            avg_acc = self.get_config_instance_and_evaluate(my_instance)
            error_rate = 1 - avg_acc
            error_rate = to_numpy(error_rate)
            return float(error_rate)

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

    def strategies(self):
        if self.search_strategy == "hyperopt":
            self.hyper_opt_strategy()
        else:
            raise NotImplementedError

    def search(self):
        self.build_search_space()
        best = self.hyper_opt_strategy()
        output_file = self.search_strategy_config["output_file"]
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            output_file = os.path.join(self.save_dir, output_file)
            with open(output_file, "w") as f:
                toml.dump(best, f)
            logger.info("Search result saved to {}".format(output_file))


def search(model_name, model, data_module, search_config, save_dir):
    logger.info("Search started ... ")
    searcher = SearchQuantization(
        model_name, model, data_module, search_config, save_dir
    )
    searcher.search()
