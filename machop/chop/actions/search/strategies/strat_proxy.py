import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib
import optuna
from functools import partial
from .base import SearchStrategyBase
from chop.passes.module.analysis import calculate_avg_bits_module_analysis_pass
from torch.utils.data import DataLoader
from itertools import islice

# For compute software metrics()
from naslib.predictors.utils.pruners.predictive import find_measures
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


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


class SearchStrategyDaddyProxy(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
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

    def compute_software_metrics(self, model):
        ### Class for meta proxy
        class NeuralModel(nn.Module):
            def __init__(self, input_size):
                super(NeuralModel, self).__init__()
                self.linear1 = nn.Linear(input_size, 64)
                self.sigmoid = nn.Sigmoid()
                self.linear2 = nn.Linear(64, 128)
                self.relu = nn.ReLU()
                self.linear3 = nn.Linear(128, 1)

            def forward(self, x):
                x = self.sigmoid(x)
                x = self.relu(self.linear1(x))
                x = self.relu(self.linear2(x))
                x = self.sigmoid(self.linear3(x))
                return x

        metrics = {}
        dataloader = self.data_module.train_dataloader()
        dataload_info = ["random", len(dataloader), 10]
        model = model.model
        device = torch.device("cuda")
        model.to(device)

        # measure_names = ['epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'plain', 'snip', 'synflow', 'zen', 'params', 'flops']
        measure_names = [
            "epe_nas",
            "fisher",
            "grad_norm",
            "grasp",
            "jacov",
            "l2_norm",
            "plain",
            "snip",
            "synflow",
            "zen",
            "params",
            "flops",
        ]
        small_proxy_scores = find_measures(
            model, dataloader, dataload_info, device, F.cross_entropy, measure_names
        )

        # load meta proxy
        proxy_model = NeuralModel(len(measure_names))
        # pretrained_model_path = r'../nas_results/model_state_dict.pt'
        pretrained_model_path = r"../nas_results/meta_proxy/meta_proxy.pt"
        proxy_model.load_state_dict(torch.load(pretrained_model_path))

        # Load mean and standard deviaiton for data normalization and store in list
        file_path = r"../nas_results/proxy_mean_stddev.json"
        with open(file_path, "r") as file:
            proxy_mean_stddev = json.load(file)

        # z normalize data and store them in a list
        measure_values_list = []
        for name in measure_names:
            val = small_proxy_scores[name]
            mean = proxy_mean_stddev[name]["mean"]
            stddev = proxy_mean_stddev[name]["stddev"]
            val_norm = (val - mean) / stddev
            measure_values_list.append(val_norm)

        measure_values_tensor = torch.tensor(measure_values_list, dtype=torch.float)

        ### Make prediction using the meta proxy
        proxy_model.eval()
        with torch.no_grad():
            prediction = proxy_model(measure_values_tensor)

        prediction_numpy = prediction.numpy()
        metrics["accuracy"] = prediction_numpy
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
            sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)

        is_eval_mode = self.config.get("eval_mode", True)
        model = search_space.rebuild_model(sampled_config, is_eval_mode)

        # build dataloader

        software_metrics = self.compute_software_metrics(
            model  # , sampled_config, is_eval_mode
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

        trial.set_user_attr("software_metrics", software_metrics)
        trial.set_user_attr("hardware_metrics", hardware_metrics)
        trial.set_user_attr("scaled_metrics", scaled_metrics)
        trial.set_user_attr("sampled_config", sampled_config)

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
