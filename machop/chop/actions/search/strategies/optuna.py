import optuna
import torch
import pandas as pd
import logging
from tabulate import tabulate

from functools import partial
from .base import SearchStrategyBase

from ....passes.analysis.total_bits_estimator import total_bits_module_analysis_pass

logger = logging.getLogger(__name__)


class SearchStrategyOptuna(SearchStrategyBase):
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

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool):
        # note that model can be mase_graph or nn.Module
        if is_eval_mode:
            with torch.no_grad():
                metrics = self.sw_runner(
                    self.data_loader, model, sampled_config, self.num_batches
                )
        else:
            metrics = self.sw_runner(
                self.data_loader, model, sampled_config, self.num_batches
            )
        return metrics

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool):
        if is_eval_mode:
            with torch.no_grad():
                metrics = self.hw_runner(
                    self.data_loader, model, sampled_config, self.num_batches
                )
        else:
            metrics = self.hw_runner(
                self.data_loader, model, sampled_config, self.num_batches
            )
        return metrics

    def objective(self, trial: optuna.trial.Trial, search_space):
        sampled_indexes = {}
        for name, length in search_space.choice_lengths_flattened.items():
            sampled_indexes[name] = trial.suggest_int(name, 0, length - 1)
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

        trial.set_user_attr("software_metrics", software_metrics)
        trial.set_user_attr("hardware_metrics", hardware_metrics)
        trial.set_user_attr("scaled_metrics", scaled_metrics)
        trial.set_user_attr("sampled_config", sampled_config)

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

        study = optuna.create_study(**study_kwargs)

        study.optimize(
            func=partial(self.objective, search_space=search_space),
            n_jobs=self.config["setup"]["n_jobs"],
            n_trials=self.config["setup"]["n_trials"],
            timeout=self.config["setup"]["timeout"],
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
        df_truncated.loc[
            :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
        ] = df_truncated.loc[
            :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
        ].applymap(
            lambda x: {k: round(v, 3) for k, v in x.items()}
        )
        txt += tabulate(
            df_truncated,
            headers="keys",
            tablefmt="orgtbl",
        )
        logger.info(f"Best trial(s):\n{txt}")
        return df
