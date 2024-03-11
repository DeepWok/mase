import optuna
import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib
import pdb

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


class SearchStrategyNaslib(SearchStrategyBase):
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
            case "bruteforce":
                sampler = optuna.samplers.BruteForceSampler()
            case _:
                raise ValueError(f"Unknown sampler name: {name}")
        return sampler

    # def objective(trial):
    #     # Suggesting weights for each metric
    #     weight_epe_nas = trial.suggest_float("weight_epe_nas", -50, 50)
    #     weight_fisher = trial.suggest_float("weight_fisher", -50, 50)
    #     weight_grad_norm = trial.suggest_float("weight_grad_norm", -50, 50)
        
    #     total_loss = 0
    #     penalty = 0  # Initialize penalty
        
    #     for item in combined_data_list:
    #         # Calculate predicted accuracy based on weights and zc metrics
    #         predicted_accuracy = (item['metrics']['epe_nas'] * weight_epe_nas +
    #                             item['metrics']['fisher'] * weight_fisher +
    #                             item['metrics']['grad_norm'] * weight_grad_norm)
            
    #         # Assume 'train_accuracy' is your target
    #         actual_accuracy = item['train_accuracy']
            
    #         # Calculating squared error loss
    #         loss = (predicted_accuracy - actual_accuracy) ** 2
    #         total_loss += loss
            
    #         # Apply penalties for predictions outside the [0, 100] range
    #         if predicted_accuracy > 100:
    #             penalty += (predicted_accuracy - 100) ** 2  # Penalty for exceeding 100%
    #         elif predicted_accuracy < 0:
    #             penalty += (predicted_accuracy - 0) ** 2  # Penalty for going below 0%
        
    #     # Incorporate penalties into the average loss
    #     average_loss_with_penalty = (total_loss + penalty) / len(combined_data_list)
        
    #     return average_loss_with_penalty


    # def objective(self, trial: optuna.trial.Trial, search_space):
    #     sampled_indexes = {}
    #     if hasattr(search_space, "optuna_sampler"):
    #         sampled_config = search_space.optuna_sampler(trial)
    #     else:
    #         for name, length in search_space.choice_lengths_flattened.items():
    #             sampled_indexes[name] = trial.suggest_int(name, 0, length - 1)
    #         sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)

    #     is_eval_mode = self.config.get("eval_mode", True)
    #     model = search_space.rebuild_model(sampled_config, is_eval_mode)

    #     software_metrics = self.compute_software_metrics(
    #         model, sampled_config, is_eval_mode
    #     )
    #     hardware_metrics = self.compute_hardware_metrics(
    #         model, sampled_config, is_eval_mode
    #     )
    #     metrics = software_metrics | hardware_metrics
    #     scaled_metrics = {}
    #     for metric_name in self.metric_names:
    #         scaled_metrics[metric_name] = (
    #             self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
    #         )

    #     trial.set_user_attr("software_metrics", software_metrics)
    #     trial.set_user_attr("hardware_metrics", hardware_metrics)
    #     trial.set_user_attr("scaled_metrics", scaled_metrics)
    #     trial.set_user_attr("sampled_config", sampled_config)

    #     self.visualizer.log_metrics(metrics=scaled_metrics, step=trial.number)

    #     if not self.sum_scaled_metrics:
    #         return list(scaled_metrics.values())
    #     else:
    #         return sum(scaled_metrics.values())

    def search(self, search_space) -> optuna.study.Study:
        print("search_space:  ", search_space)
        
        # pdb.set_trace()
        study_kwargs = {
            "sampler": self.sampler_map(self.config["setup"]["sampler"]),
        }
        # if not self.sum_scaled_metrics:
        #     study_kwargs["directions"] = self.directions
        # else:
        #     study_kwargs["direction"] = self.direction

        # if isinstance(self.config["setup"].get("pkl_ckpt", None), str):
        #     study = joblib.load(self.config["setup"]["pkl_ckpt"])
        #     logger.info(f"Loaded study from {self.config['setup']['pkl_ckpt']}")
        # else:
        
        study = optuna.create_study(**study_kwargs)

        # study.optimize(
        #     func=partial(self.objective, search_space=search_space),
        #     n_jobs=self.config["setup"]["n_jobs"],
        #     n_trials=self.config["setup"]["n_trials"],
        #     timeout=self.config["setup"]["timeout"],
        #     callbacks=[
        #         partial(
        #             callback_save_study,
        #             save_dir=self.save_dir,
        #             save_every_n_trials=self.config["setup"].get(
        #                 "save_every_n_trials", 10
        #             ),
        #         )
        #     ],
        #     show_progress_bar=True,
        # )

        self._save_study(study, self.save_dir / "study.pkl")
        self._save_search_dataframe(study, search_space, self.save_dir / "log.json")
        # self._save_best(study, self.save_dir / "best.json")

        self._save_best_zero_cost(search_space, self.save_dir / "metrics.json")

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
    def _save_best_zero_cost(search_space, save_path):
        # import pdb
        # pdb.set_trace()

        result_dict = {}
        for item in search_space.zcp_results:
            key = list(item.keys())[0]
            values = item[key]
            result_dict[key] = {'test_spearman': values['test_spearman'], 'train_spearman': values['train_spearman']}


        print("result_dict:  ", result_dict)


        df = pd.DataFrame(
            columns=[
                "number",
                "result_dict",
            ]
        )
        row = [
            0,
            result_dict,
            ]
        
        df.loc[len(df)] = row
        # df.to_json(save_path, orient="index", indent=4)

        txt = "Best trial(s):\n"
        print("df:  ", df)
        df_truncated = df.loc[
            :, ["result_dict"]
        ]

        def beautify_metric(metric: dict):
            beautified = {}
            for k, v in metric.items():
                if isinstance(v, (float, int)):
                    beautified[k] = round(v, 3)
                else:
                    txt = str(v)
                    if len(txt) > 40:
                        txt = txt[:40] + "..."
                    else:
                        txt = txt[:40]
                    beautified[k] = txt
            return beautified

        df_truncated.loc[
            :, ["result_dict"]
        ] = df_truncated.loc[
            :, ["result_dict"]
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
