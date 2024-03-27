import optuna
import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib
from ..search_space.zero_cost_proxy.utils import evaluate_predictions

from functools import partial
from .base import SearchStrategyBase

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


class SearchStrategyZeroCost(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
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

    def combined_list(self, data):
        # Initialize a dictionary to hold combined data
        combined_data = {}

        # Iterate through each metric's data
        for metric_data in data:
            for metric_name, details in metric_data.items():
                for result in details["results"]:
                    # Convert the test_hash tuple to a string to use it as a dictionary key
                    test_hash_key = str(result["test_hash"])

                    if test_hash_key not in combined_data:
                        combined_data[test_hash_key] = {
                            "test_hash": result["test_hash"],
                            "test_accuracy": result["test_accuracy"],
                            "metrics": {},
                        }

                    # Store the zc_metric under the corresponding metric name
                    combined_data[test_hash_key]["metrics"][metric_name] = result[
                        "zc_metric"
                    ]

        # Convert the combined data back to a list format if needed
        combined_data_list = list(combined_data.values())

        return combined_data_list

    def objective(self, trial, search_space):
        combined_data_list = self.combined_list(search_space.zcp_results)

        # Dynamically find all unique metric names
        unique_metric_names = set()
        for item in combined_data_list:
            unique_metric_names.update(item["metrics"].keys())

        # Convert set to list to ensure consistent ordering
        unique_metric_names = sorted(list(unique_metric_names))

        # Suggest weights for each unique metric dynamically
        weights = {
            metric: trial.suggest_float(
                f"{metric}",
                self.config["setup"]["weight_lower_limit"],
                self.config["setup"]["weight_upper_limit"],
            )
            for metric in unique_metric_names
        }

        total_loss = 0
        penalty = 0

        for item in combined_data_list:
            # Dynamically calculate predicted accuracy based on the weights of the zc metrics
            predicted_accuracy = sum(
                item["metrics"][metric] * weights[metric]
                for metric in item["metrics"].keys()
            )

            # Assume 'test_accuracy' is your target
            actual_accuracy = item["test_accuracy"]

            # Calculating squared error loss
            loss = (predicted_accuracy - actual_accuracy) ** 2
            total_loss += loss

            # Apply penalties for predictions outside the [0, 100] range
            if predicted_accuracy > 100:
                penalty += (predicted_accuracy - 100) ** 2
            elif predicted_accuracy < 0:
                penalty += (predicted_accuracy) ** 2

        # Incorporate penalties into the average loss
        average_loss_with_penalty = (total_loss + penalty) / len(combined_data_list)

        return average_loss_with_penalty

    def get_optuna_prediction(self, model_results, best_params):
        """
        Calculates the ensemble weight for a given model result and the best parameters.

        The ensemble weight is computed by summing the product of each metric's value
        in the model result with its corresponding best parameter value.

        Args:
            model_result (dict): A dictionary containing the model's results, including a 'metrics' sub-dictionary.
            best_params (dict): A dictionary containing the best parameter value for each metric.

        Returns:
            float: The calculated ensemble weight.
        """

        for x in model_results:
            x["metrics"]["optuna_ensemble"] = sum(
                x["metrics"][metric] * best_params[metric] for metric in x["metrics"]
            )

        return model_results

    def search(self, search_space) -> optuna.study.Study:
        study_kwargs = {
            "sampler": self.sampler_map(self.config["setup"]["sampler"]),
            "direction": self.config["setup"]["direction"],
        }

        study = optuna.create_study(**study_kwargs)

        study.optimize(
            func=partial(self.objective, search_space=search_space),
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

        best_params = study.best_params

        model_results = self.combined_list(search_space.zcp_results)
        model_results = self.get_optuna_prediction(model_results, best_params)

        self._save_study(study, self.save_dir / "study.pkl")
        self._save_search_dataframe(study, self.save_dir / "log.json")
        self._save_best_zero_cost(
            search_space, model_results, self.save_dir / "metrics.json"
        )

        return study

    @staticmethod
    def _save_search_dataframe(study: optuna.study.Study, save_path):
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
    def _save_best_zero_cost(search_space, model_results, save_path):
        # calculate ensemble metric
        ytest = [x["test_accuracy"] for x in model_results]
        ensemble_preds = [x["metrics"]["optuna_ensemble"] for x in model_results]
        ensemble_metric = evaluate_predictions(ytest, ensemble_preds)

        result_dict = {
            "optuna_ensemble_metric": {
                "test_spearman": ensemble_metric["spearmanr"],
                "test_kendaltau": ensemble_metric["kendalltau"],
            },
            search_space.config["zc"]["ensemble_model"]: {
                "test_spearman": search_space.custom_ensemble_metrics["spearmanr"],
                "test_kendaltau": search_space.custom_ensemble_metrics["kendalltau"],
            },
            "xgboost": {
                "test_spearman": search_space.xgboost_metrics["spearmanr"],
                "test_kendaltau": search_space.xgboost_metrics["kendalltau"],
            },
        }
        for item in search_space.zcp_results:
            key = list(item.keys())[0]
            values = item[key]
            result_dict[key] = {
                "test_spearman": values["test_spearman"],
                "train_spearman": values["train_spearman"],
                "test_kendaltau": values["test_kendaltau"],
            }

        # save to metrics.json
        save_df = pd.DataFrame(columns=["number", "result_dict", "model_results"])
        row = [
            0,
            result_dict,
            model_results,
        ]

        save_df.loc[len(save_df)] = row
        save_df.to_json(save_path, orient="index", indent=4)

        # List of custom dictionaries for "Global Parameters"
        dict_list = [
            {"num_training_archs": search_space.config["zc"]["num_archs_train"]},
            {"num_testing_archs": search_space.config["zc"]["num_archs_test"]},
            {"dataset": search_space.config["zc"]["dataset"]},
            {"benchmark": search_space.config["zc"]["benchmark"]},
            {"num_zc_proxies": len(search_space.config["zc"]["zc_proxies"])},
        ]

        # Initialize the DataFrame with an empty column list if it's solely for the results below
        df = pd.DataFrame()
        df["spearman"] = [
            {key: value["test_spearman"]}
            for key, value in sorted(
                result_dict.items(),
                key=lambda item: item[1]["test_spearman"],
                reverse=True,
            )
        ]
        df["kendaltau"] = [
            {key: value["test_kendaltau"]}
            for key, value in sorted(
                result_dict.items(),
                key=lambda item: item[1]["test_kendaltau"],
                reverse=True,
            )
        ]
        df["Global Parameters"] = pd.Series(dict_list[: len(df)])

        txt = "Best trial(s):\n"
        df_truncated = df.loc[:, ["spearman", "kendaltau", "Global Parameters"]].head()

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
            :, ["spearman", "kendaltau", "Global Parameters"]
        ] = df_truncated.loc[:, ["spearman", "kendaltau", "Global Parameters"]].map(
            beautify_metric
        )
        txt += tabulate(
            df_truncated,
            headers="keys",
            tablefmt="orgtbl",
        )
        logger.info(f"Best trial(s):\n{txt}")
        return df