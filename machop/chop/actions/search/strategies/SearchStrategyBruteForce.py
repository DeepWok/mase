import optuna
import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib
import numpy as np

from functools import partial
from .base import SearchStrategyBase

from chop.passes.module.analysis import calculate_avg_bits_module_analysis_pass

logger = logging.getLogger(__name__)

def brute_force_combinations(variable_ranges, current_combination=[], all_combinations=[]):
    if not variable_ranges:
        all_combinations.append(current_combination)
        return
    for value in variable_ranges[0]:
        brute_force_combinations(variable_ranges[1:], current_combination + [value], all_combinations)
    return all_combinations

def lists_to_dict(keys, values):
    return dict(zip(keys, values))

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


class SearchStrategyBruteForce(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))

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

    def objective(self, sampled_index, search_space):

        sampled_config = search_space.flattened_indexes_to_config(sampled_index)
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
        scaled_metrics_aggregrate = 0
        for metric_name in self.metric_names:
            scaled_metrics[metric_name] = (
                self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
            )

            if not self.sum_scaled_metrics:
                if self.config["metrics"][metric_name]["direction"] == "minimize":   
                    scaled_metrics_aggregrate -= scaled_metrics[metric_name] 
                else:
                    scaled_metrics_aggregrate += scaled_metrics[metric_name]   
            else:
                if self.config["setup"]["direction"] == "minimize":    
                    scaled_metrics_aggregrate -= scaled_metrics[metric_name] 
                else:
                    scaled_metrics_aggregrate += scaled_metrics[metric_name]   
            

        self.visualizer.log_metrics(metrics=scaled_metrics, step=i)

    
        return metrics, scaled_metrics, scaled_metrics_aggregrate 
        
    def search(self, search_space) -> optuna.study.Study:
        keys = search_space.choice_lengths_flattened.keys()
        keys_list = list(keys)

        variable_ranges = []
        no_combinations = 1
        for k,v in search_space.choice_lengths_flattened.items():
            variable_ranges.append(list(range(v)))
            no_combinations = no_combinations*v

        all_combinations = brute_force_combinations(variable_ranges)
        metrics=[]
        scaled_metrics=[]
        scaled_metrics_aggregrate=[]
        for i in range(no_combinations):
            sampled_index = lists_to_dict(keys_list, all_combinations[i])
            print(sampled_index)
            
            metrics[i],scaled_metrics[i],scaled_metrics_aggregrate[i] = objective(self, sampled_index, search_space)

        best_index = np.argmax(scaled_metrics_aggregrate)
        best_metrics = metrics[best_index]  # Parameters corresponding to best result
        best_scaled_metric = scaled_metrics[best_index]  # Metric A value for best result
        best_config = search_space.flattened_indexes_to_config(lists_to_dict(keys_list, all_combinations[best_index]))
        search_result = {"index":best_index,"metrics":best_metrics,"scaled metrics":best_scaled_metric,"config": best_config}
        self._save_study(search_result, self.save_dir / "best_result")
        return search_result

    @staticmethod
    def _save_best(search_result, save_path):
        df = pd.DataFrame(
            columns=[
                "number",
                "value",
                "software_metrics",
                "hardware_metrics",
                "scaled_metrics",
                "sampled_config",
            ]
        )
        row = [
                trial.number,
                trial.values,
                trial.user_attrs["software_metrics"],
                trial.user_attrs["hardware_metrics"],
                trial.user_attrs["scaled_metrics"],
                trial.user_attrs["sampled_config"],
            ]
        df.loc[len(df)] = row
       
        df.to_json(save_path, orient="index", indent=4)

        txt = "Best trial(s):\n"
        df_truncated = df.loc[
            :, ["number", "software_metrics", "hardware_metrics", "scaled_metrics"]
        ]

        def beautify_metric(metric: dict):
            beautified = {}
            for k, v in metric.items():
                if isinstance(v, (float, int)):
                    beautified[k] = round(v, 3)
                else:
                    txt = str(v)
                    if len(txt) > 20:
                        txt = txt[:20] + "..."
                    else:
                        txt = txt[:20]
                    beautified[k] = txt
            return beautified

        df_truncated.loc[
            :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
        ] = df_truncated.loc[
            :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
        ].map(
            beautify_metric
        )
        txt += tabulate(
            df_truncated,
            headers="keys",
            tablefmt="orgtbl",
        )
        logger.info(f"Best trial(s):\n{txt}")
        return df