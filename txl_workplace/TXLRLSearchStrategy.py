import optuna
import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib

from functools import partial
from chop.actions.search.strategies.base import SearchStrategyBase

from chop.passes.module.analysis import calculate_avg_bits_module_analysis_pass

logger = logging.getLogger(__name__)


def callback_save_study(
    study: optuna.study.Study,
    frozen_trial: optuna.trial.FrozenTrial,
    save_dir,
    save_every_n_trials=1,
):
    if (frozen_trial.number + 1) % save_every_n_trials == 0:
        study_path = save_dir / f"study_trial_{frozen_trial.number}.pkl"
        if not study_path.parent.exists():
            study_path.parent.mkdir(parents=True)

        with open(study_path, "wb") as f:
            joblib.dump(study, f)


class SearchStrategyRL(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        # self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        self.bf_result = {}
        self.best_config = None
        self.highest_score = None
        self.counter = 0
        # if not self.sum_scaled_metrics:
        #     self.directions = [
        #         self.config["metrics"][k]["direction"] for k in self.metric_names
        #     ]
        # else:
        #     self.direction = self.config["setup"]["direction"]

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool):
        # note that model can be mase_graph or nn.Module
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.sw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.sw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool):
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics

    def search(self, search_space):
        sampled_indexes = {}
        name_list = list(search_space.choice_lengths_flattened.keys())
        # TODO replace dynamic_bf to model learn in stable baseline3
        # Complexity of Bruce-Force is Cartesian product of the numbers of possible choices of each component
        # using recursion to implement dynamic components Bruce-Force
        def dynamic_bf(index = 0):
            if index == len(name_list):
                score = self.run_trial(search_space, sampled_indexes)
                self.counter += 1
                print(f'For proposal-{self.counter}, score: {score}')
                # print(score)
                self.bf_result[str(sampled_indexes)] = score
                if self.highest_score is None or score > self.highest_score:
                    self.highest_score = score
                    self.best_config = sampled_indexes
                return
            for i in range(search_space.choice_lengths_flattened[name_list[index]]):
                sampled_indexes[name_list[index]] = i
                dynamic_bf(index + 1)
        dynamic_bf()
        self.counter = 0
        # print(self.bf_result)
        print("best config: ", self.best_config)
        print("highest score: ", self.highest_score)

    def run_trial(self, search_space, sampled_indexes):
        sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)

        is_eval_mode = self.config.get("eval_mode", True)
        model = search_space.rebuild_model(sampled_config, is_eval_mode)
        software_metrics = self.compute_software_metrics(
            model, sampled_config, is_eval_mode
        )
        hardware_metrics = self.compute_hardware_metrics(
            model, sampled_config, is_eval_mode
        )
        metrics = software_metrics | hardware_metrics
        scaled_metrics = {}
        for metric_name in self.metric_names:
            scaled_metrics[metric_name] = (
                    self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
            )
        return sum(scaled_metrics.values())