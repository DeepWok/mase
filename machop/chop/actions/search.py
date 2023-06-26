import copy
import logging
import operator
import os
import random
from functools import partial

import joblib
import optuna
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperopt import fmin, hp, rand, space_eval, tpe
from machop.modify.modifier import Modifier, is_modifiable
from machop.utils import to_numpy
from tqdm import tqdm

from ..graph.mase_tracer import mase_symbolic_trace
from ..graph.utils import get_module_by_target
from .plt_wrapper import get_model_wrapper

logger = logging.getLogger(__name__)


class SearchSpace:
    def __init__(
        self,
        model_name,
        model,
        is_nlp_model,
        task,
        info,
        modifier_kwargs,
        data_module,
        search_config,
        save_dir,
        accelerator="auto",
    ) -> None:
        self.model_name = model_name
        self.info = info
        if is_nlp_model:
            self.blueprint_model = {
                k: v.to("cpu") if isinstance(v, nn.Module) else v
                for k, v in model.items()
            }
        else:
            self.blueprint_model = model.to("cpu")
        self.is_nlp_model = is_nlp_model
        self.task = task
        self.modifier_kwargs = modifier_kwargs
        self._parse_config(search_config)
        self._prepare_loader(data_module)
        self._set_accelerator(accelerator)
        self.graph_module = mase_symbolic_trace(
            model["model"] if is_nlp_model else model,
            concrete_args=modifier_kwargs["dummy_inputs_for_fx"],
        )
        self.save_dir = save_dir
        self._create_logger()

    def _create_logger(self):
        logger = logging.getLogger("Search-sw")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.save_dir, "searched-values.log"))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        self.logger = logger

    def _parse_config(self, search_config):
        with open(search_config, "r") as f:
            search_args = toml.load(f)
        # building search space
        self.search_strategy_config = search_args["strategy"]
        self.search_data_config = search_args["data"]
        self.seed_search_space = search_args["seed_search_space"]
        self.nodes_to_ignore = search_args["nodes_to_ignore"]
        self.bitwidth_constraint = search_args["bitwidth_constraint"]
        assert self.search_data_config["data_loader"] in [
            "train_dataloader",
            "val_dataloader",
            "test_dataloader",
        ]

    def _prepare_loader(self, data_module):
        data_module.prepare_data()
        data_module.setup()

        self.data_loader = getattr(
            data_module, self.search_data_config["data_loader"]
        )()
        self.num_batches = self.search_data_config["num_batches"]

    def _set_accelerator(self, accelerator):
        if accelerator == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif accelerator == "gpu":
            self.device = torch.device("cuda:0")
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise RuntimeError(f"Unsupported accelerator {accelerator}")

    def rebuild_model(self, sampled_config):
        model = copy.deepcopy(self.blueprint_model)
        modifier = Modifier(
            model=model["model"] if self.is_nlp_model else model,
            config_path=sampled_config,
            silent=True,
            **self.modifier_kwargs,
        )
        return modifier.modify()

    def search(self):
        raise NotImplementedError


class SearchQuantization(SearchSpace):
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

        for n in self.graph_module.graph.nodes:
            if n.op not in ["call_function", "call_module", "call_method"]:
                continue
            node_name = n.name if n.op in ["call_function", "call_method"] else n.target
            if node_name in self.nodes_to_ignore:
                continue
            if not is_modifiable(
                n,
                self.blueprint_model["model"]
                if self.is_nlp_model
                else self.blueprint_model,
            ):
                continue

            if n.op == "call_function":
                function_nodes_to_modify[
                    n.name
                ] = self.create_function_search_space_from_seed(n.target)
            elif n.op == "call_module":
                real_target = get_module_by_target(
                    self.blueprint_model["model"]
                    if self.is_nlp_model
                    else self.blueprint_model,
                    n.target,
                )
                module_nodes_to_modify[
                    n.target
                ] = self.create_module_search_space_from_seed(real_target)
            elif n.op == "call_method":
                method_nodes_to_modify[
                    n.name
                ] = self.create_method_search_space_from_seed(n.target)

        self.complete_search_space = {
            "function_nodes_to_modify": function_nodes_to_modify,
            "module_nodes_to_modify": module_nodes_to_modify,
            "method_nodes_to_modify": method_nodes_to_modify,
        }

    def create_module_search_space_from_seed(self, real_target):
        seed_sp = self.seed_search_space
        if isinstance(real_target, (nn.Linear, nn.Conv2d)):
            config = copy.copy(seed_sp)
        elif isinstance(real_target, (nn.ReLU,)):
            config = {
                "name": seed_sp["name"],
                "data_in_width": seed_sp["data_in_width"],
                "data_in_frac_width": seed_sp["data_in_frac_width"],
            }
        else:
            logger.warning(f"Unrecognized module {real_target}")
            config = copy.copy(seed_sp)

        return config

    def create_function_search_space_from_seed(self, real_target):
        seed_sp = self.seed_search_space
        if real_target in (F.relu,):
            config = {
                "name": seed_sp["name"],
                "data_in_width": seed_sp["data_in_width"],
                "data_in_frac_width": seed_sp["data_in_frac_width"],
            }
        elif real_target in (torch.add, operator.add):
            config = {
                "name": seed_sp["name"],
                "data_in_width": seed_sp["data_in_width"],
                "data_in_frac_width": seed_sp["data_in_frac_width"],
            }
        elif real_target in (torch.matmul, torch.bmm):
            config = {
                "name": seed_sp["name"],
                "data_in_width": seed_sp["data_in_width"],
                "data_in_frac_width": seed_sp["data_in_frac_width"],
                "weight_width": seed_sp["weight_width"],
                "weight_frac_width": seed_sp["weight_frac_width"],
            }
        else:
            logger.warning(f"Unrecognized function {real_target}")
            config = copy.copy(seed_sp)

        return config

    def create_method_search_space_from_seed(self, real_target):
        seed_sp = self.seed_search_space
        if real_target in ("bmm", "matmul"):
            config = {
                "name": seed_sp["name"],
                "data_in_width": seed_sp["data_in_width"],
                "data_in_frac_width": seed_sp["data_in_frac_width"],
                "weight_width": seed_sp["weight_width"],
                "weight_frac_width": seed_sp["weight_frac_width"],
            }
        else:
            logger.warning(f"Unrecognized method {real_target}")
            config = copy.copy(seed_sp)
        return config

    def compute_loss(self, new_model):
        wrapped_model = get_model_wrapper(self.model_name, task=self.task)(
            new_model, info=self.info
        ).to(self.device)
        wrapped_model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                if batch_idx >= self.num_batches:
                    break

                batch = self._move_batch_to_device(batch)
                loss = wrapped_model.validation_step(batch, batch_idx)
                losses.append(torch.mean(loss).item())
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

    def compute_bitwidth(self, sampled_config):
        bitwidth = 0
        for node_name, sub_config in sampled_config["function_nodes_to_modify"].items():
            bitwidth += self.get_bits_from_config(sub_config)
        for node_name, sub_config in sampled_config["module_nodes_to_modify"].items():
            bitwidth += self.get_bits_from_config(sampled_config)
        for node_name, sub_config in sampled_config["method_nodes_to_modify"].items():
            bitwidth += self.get_bits_from_config(sampled_config)
        return bitwidth

    def get_bits_from_config(self, sub_config):
        bitwidth = 0
        if "data_in_width" in sub_config:
            bitwidth += sub_config["data_in_width"]
        if "weight_width" in sub_config:
            bitwidth += sub_config["weight_width"]
        if "bias_width" in sub_config:
            bitwidth += sub_config["bias_width"]
        return bitwidth

    def compute_objective(self, sampled_config):
        new_model = self.rebuild_model(sampled_config)
        if self.is_nlp_model:
            model = copy.deepcopy(self.blueprint_model)
            model.update(model=new_model)
            new_model = model

        acc_loss = self.compute_loss(new_model)
        bitwidth_loss = 0
        if self.bitwidth_constraint["enable"]:
            total_bits = self.compute_bitwidth(sampled_config)
            bitwidth_loss = self.bitwidth_constraint["weight"] * total_bits

        return acc_loss, bitwidth_loss

    def search(self):
        self.build_search_space()

        def sample_a_config(trail: optuna.Trial):
            sampled_config = {}
            for meta_name, meta_value in self.complete_search_space.items():
                sampled_config[meta_name] = {}
                for node, sub_config in meta_value.items():
                    sampled_config[meta_name][node] = {}
                    for key, values in sub_config.items():
                        sampled_config[meta_name][node][
                            key
                        ] = trail.suggest_categorical(
                            f"{meta_name}::{node}::{key}", values
                        )

            return sampled_config

        def objective(trial):
            sampled_config = sample_a_config(trial)
            sampled_config["default"] = copy.copy(self.default_config)
            acc_loss, bitwidth_loss = self.compute_objective(sampled_config)
            return acc_loss, bitwidth_loss

        def logger_callback(study, frozen_trial):
            self.logger.info(
                "Trial {} done. (acc loss, bit loss): {}".format(
                    frozen_trial.number, frozen_trial.values
                )
            )

#             study_path = os.path.join(self.save_dir, "study.pkl")
#             joblib.dump(study, study_path)

        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(
            # partial(objective, complete_search_space=self.complete_search_space),
            objective,
            n_trials=self.search_strategy_config["n_trials"],
            n_jobs=self.search_strategy_config.get("n_jobs", 1),
            show_progress_bar=True,
            callbacks=[logger_callback],
        )
        return study

    def save_study_and_config(self):
        study: optuna.Study = self.search()

        logger.info("=== Best trials ===")
        self.logger.info("=== Best trials ===")
        for i, trial in enumerate(study.best_trials):
            q_config_save_path = os.path.join(self.save_dir, f"best_{i}_q_config.toml")
            SearchQuantization.save_a_trial_to_q_config(trial, q_config_save_path)
            self.logger.info(
                "Trial {} done. (acc loss, bit loss) : {}, q_config saved to {}".format(
                    trial.number, trial.values, q_config_save_path
                )
            )
            logger.info(
                "Trial {} done. (acc loss, bit loss): {}, q_config saved to {}".format(
                    trial.number, trial.values, q_config_save_path
                )
            )

        study_path = os.path.join(self.save_dir, "study.pkl")
        joblib.dump(study, study_path)
        logger.info("Study saved to {}".format(os.path.abspath(study_path)))
        logger.info(
            "Optization logs saved to {}".format(
                os.path.abspath(os.path.join(self.save_dir, "searched-values.log"))
            )
        )

    @classmethod
    def save_a_trial_to_q_config(cls, trial: optuna.Trial, save_path):
        q_config = {
            "default": copy.copy(cls.default_config),
            "function_nodes_to_modify": {},
            "module_nodes_to_modify": {},
            "method_nodes_to_modify": {},
        }

        params = trial.params
        for param_name, width in params.items():
            meta, node, key = param_name.split("::")
            if node not in q_config[meta]:
                q_config[meta][node] = {key: width}
            else:
                q_config[meta][node][key] = width

        with open(save_path, "w+") as f:
            toml.dump(q_config, f)


def search(
    model_name,
    model,
    is_nlp_model,
    task,
    info,
    modifier_kwargs,
    data_module,
    search_config,
    save_dir,
    accelerator="auto",
):
    logger.info("Search started...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    searcher = SearchQuantization(
        model_name=model_name,
        model=model,
        is_nlp_model=is_nlp_model,
        task=task,
        info=info,
        modifier_kwargs=modifier_kwargs,
        data_module=data_module,
        search_config=search_config,
        save_dir=save_dir,
        accelerator=accelerator,
    )
    searcher.search()
    searcher.save_study_and_config()
    logger.info("Search finished.")
