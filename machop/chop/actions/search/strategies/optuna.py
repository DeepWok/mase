import optuna
from sklearn.impute import SimpleImputer
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
from tabulate import tabulate
import joblib

from functools import partial
from .base import SearchStrategyBase

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


class SearchStrategyOptuna(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))

        ### whether it is zero cost search
        self.zero_cost_mode = "zero_cost" in self.config["sw_runner"].keys()
        if self.zero_cost_mode:
            ### dict for saving the proxy number and the true metric
            self.zero_cost_and_true_metric = {}

        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]

    def sampler_map(self, name):
        match name.lower():
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "nsgaii":
                sampler = optuna.samplers.NSGAIISampler()
            case "nsgaiii":
                sampler = optuna.samplers.NSGAIIISampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case _:
                raise ValueError(f"Unknown sampler name: {name}")
        return sampler

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

    def objective(self, trial: optuna.trial.Trial, search_space):
        sampled_indexes = {}
        if hasattr(search_space, "optuna_sampler"):
            sampled_config = search_space.optuna_sampler(trial)
        else:
            for name, length in search_space.choice_lengths_flattened.items():
                sampled_indexes[name] = trial.suggest_int(name, 0, length - 1)
            
            trial_params = trial.params.popitem()
            sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)

        is_eval_mode = self.config.get("eval_mode", True)
        
        '''
        group 2: zero cost

        In this section, we've added functionality to handle the zero cost mode. 

        - First, we check if the zero cost mode is enabled.
        - If it is, we rebuild the model with the sampled configuration and evaluation mode, and store the data returned by the rebuild_model function.
        - We then create a new entry in the zero_cost_and_true_metric dictionary with the true metric data and an empty dictionary for the zero cost proxy.
        - If the zero cost mode is not enabled, we simply rebuild the model without storing the data.
        - We then compute the software and hardware metrics, and combine them into a single dictionary.
        - We scale the metrics according to the scale factors defined in the configuration.
        - If the zero cost mode is enabled, we store the scaled metrics in the zero_cost_proxy dictionary.
        - We then set several user attributes for the trial, including the software metrics, hardware metrics, scaled metrics, sampled configuration, and nasbench data metrics.
        - We log the scaled metrics using the visualizer.
        - Finally, we return the scaled metrics. If the sum_scaled_metrics flag is enabled, we return the sum of the scaled metrics; otherwise, we return the list of scaled metrics.
        '''
        
        # print("self config")
        # print(self.config)
        if self.zero_cost_mode:
            model, data = search_space.rebuild_model(sampled_config, is_eval_mode)
            length = len(self.zero_cost_and_true_metric)
            self.zero_cost_and_true_metric[length] = {
                "true_metric": data,
                "zero_cost_proxy": {}
            }
        else:
            model = search_space.rebuild_model(sampled_config, is_eval_mode)
            data = None
        # print("data")
        # print(data)

        software_metrics = self.compute_software_metrics(
            model, sampled_config, is_eval_mode
        )
        hardware_metrics = self.compute_hardware_metrics(
            model, sampled_config, is_eval_mode
        )
        metrics = software_metrics | hardware_metrics
        # print("Overall metrics: ")
        # print(metrics)
        scaled_metrics = {}
        for metric_name in self.metric_names:
            scaled_metrics[metric_name] = (
                self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
            )
        if self.zero_cost_mode:
            self.zero_cost_and_true_metric[length]["zero_cost_proxy"] = scaled_metrics

        trial.set_user_attr("software_metrics", software_metrics)
        trial.set_user_attr("hardware_metrics", hardware_metrics)
        trial.set_user_attr("scaled_metrics", scaled_metrics)
        trial.set_user_attr("sampled_config", sampled_config)
        trial.set_user_attr("nasbench_data_metrics", data)

        self.visualizer.log_metrics(metrics=scaled_metrics, step=trial.number)

        if not self.sum_scaled_metrics:
            return list(scaled_metrics.values())
        else:
            return sum(scaled_metrics.values())

    def search(self, search_space) -> optuna.study.Study:
        study_kwargs = {
            "sampler": self.sampler_map(self.config["setup"]["sampler"]),
        }
        if not self.sum_scaled_metrics:
            study_kwargs["directions"] = self.directions
        else:
            study_kwargs["direction"] = self.direction

        if isinstance(self.config["setup"].get("pkl_ckpt", None), str):
            study = joblib.load(self.config["setup"]["pkl_ckpt"])
            logger.info(f"Loaded study from {self.config['setup']['pkl_ckpt']}")
        else:
            study = optuna.create_study(**study_kwargs)

        study.optimize(
            func=partial(self.objective, search_space=search_space),
            n_jobs=self.config["setup"]["n_jobs"],
            n_trials=self.config["setup"]["n_trials"],
            timeout=self.config["setup"]["timeout"],
            callbacks=[
                partial(
                    callback_save_study,
                    save_dir=self.save_dir,
                    save_every_n_trials=self.config["setup"].get(
                        "save_every_n_trials", 10
                    ),
                )
            ],
            show_progress_bar=True,
        )
        self._save_study(study, self.save_dir / "study.pkl")
        self._save_search_dataframe(study, search_space, self.save_dir / "log.json")
        self._save_best(study, self.save_dir / "best.json")

        return study

    def zero_cost_weight(self):
        if self.zero_cost_mode:
            self.zc_proxy = pd.DataFrame()
            self.zc_true_accuracy = []
            for key, value in self.zero_cost_and_true_metric.items() :
                self.zc_proxy = pd.concat([self.zc_proxy, pd.DataFrame(value["zero_cost_proxy"], index=[0])], ignore_index=True)
                self.zc_true_accuracy.append(value["true_metric"]["test-accuracy"])

            # get rid of hardware metrics
            available_zc_metrics = ["fisher", "grad_norm", "grasp", "l2_norm", "plain", "snip", "synflow", "naswot", "naswot_relu", "tenas", "zico"]
            zc_cols = self.zc_proxy.columns[self.zc_proxy.columns.isin(available_zc_metrics)]
            self.zc_proxy = self.zc_proxy[zc_cols]
            
            # print("proxy values")
            # print(self.zc_proxy)

            ### deal with -inf (fill with the minimum)
            # self.zc_proxy.loc[np.isneginf(self.zc_proxy["jacob_cov"]), "jacob_cov"] = min(self.zc_proxy.loc[~np.isneginf(self.zc_proxy["jacob_cov"]), "jacob_cov"])

            ### fit linear regression models
            ## standardize
            for _ in self.zc_proxy.columns:
                if self.zc_proxy[_].mean().sum() != 0:
                    self.zc_proxy[_] = (self.zc_proxy[_] - self.zc_proxy[_].mean())/self.zc_proxy[_].std()
                else:
                    self.zc_proxy[_] = (self.zc_proxy[_] - self.zc_proxy[_].mean())
                    
            # print("proxy values after standardization")
            # print(self.zc_proxy)
            
            self.zc_weight_model = LinearRegression(fit_intercept=True)
            self.zc_weight_model.fit(self.zc_proxy, self.zc_true_accuracy)
        else:
            raise ValueError("zero_cost_mode is Fasle, do not fit the zero_cost_weight evaluation.")
            

    @staticmethod
    def _save_search_dataframe(study: optuna.study.Study, search_space, save_path):
        df = study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "user_attrs",
                "system_attrs",
                "state",
                "datetime_start",
                "datetime_complete",
                "duration",
            )
        )
        df.to_json(save_path, orient="index", indent=4)
        return df

    @staticmethod
    def _save_best(study: optuna.study.Study, save_path):
        df = pd.DataFrame(
            columns=[
                "number",
                "value",
                "software_metrics",
                "hardware_metrics",
                "scaled_metrics",
                "sampled_config",
                "nasbench_data_metrics",
            ]
        )
        if study._is_multi_objective:
            best_trials = study.best_trials
            for trial in best_trials:
                row = [
                    trial.number,
                    trial.values,
                    trial.user_attrs["software_metrics"],
                    trial.user_attrs["hardware_metrics"],
                    trial.user_attrs["scaled_metrics"],
                    trial.user_attrs["sampled_config"],
                    trial.user_attrs["nasbench_data_metrics"],
                ]
                df.loc[len(df)] = row
        else:
            best_trial = study.best_trial
            row = [
                best_trial.number,
                best_trial.value,
                best_trial.user_attrs["software_metrics"],
                best_trial.user_attrs["hardware_metrics"],
                best_trial.user_attrs["scaled_metrics"],
                best_trial.user_attrs["sampled_config"],
                best_trial.user_attrs["nasbench_data_metrics"],
            ]
            df.loc[len(df)] = row
        df.to_json(save_path, orient="index", indent=4)

        txt = "Best trial(s):\n"
        df_truncated = df.loc[
            :, ["number", "software_metrics", "hardware_metrics", "scaled_metrics", "nasbench_data_metrics"]
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
            :, ["software_metrics", "hardware_metrics", "scaled_metrics", "nasbench_data_metrics"]
        ] = df_truncated.loc[
            :, ["software_metrics", "hardware_metrics", "scaled_metrics", "nasbench_data_metrics"]
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
