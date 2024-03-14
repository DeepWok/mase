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
                for result in details['results']:
                    # Convert the test_hash tuple to a string to use it as a dictionary key
                    test_hash_key = str(result['test_hash'])
                    
                    if test_hash_key not in combined_data:
                        combined_data[test_hash_key] = {
                            'test_hash': result['test_hash'],
                            'test_accuracy': result['test_accuracy'],
                            'metrics': {}
                        }
                    
                    # Store the zc_metric under the corresponding metric name
                    combined_data[test_hash_key]['metrics'][metric_name] = result['zc_metric']

        # Convert the combined data back to a list format if needed
        combined_data_list = list(combined_data.values())

        return combined_data_list
    
    def objective(self, trial, search_space):
        combined_data_list = self.combined_list(search_space.zcp_results)
    
        # Dynamically find all unique metric names
        unique_metric_names = set()
        for item in combined_data_list:
            unique_metric_names.update(item['metrics'].keys())

        # Convert set to list to ensure consistent ordering
        unique_metric_names = sorted(list(unique_metric_names))

        # Suggest weights for each unique metric dynamically
        weights = {metric: trial.suggest_float(f"{metric}", self.config["setup"]["weight_lower_limit"], self.config["setup"]["weight_upper_limit"]) for metric in unique_metric_names}

        total_loss = 0
        penalty = 0

        for item in combined_data_list:
            # Dynamically calculate predicted accuracy based on the weights of the zc metrics
            predicted_accuracy = sum(item['metrics'][metric] * weights[metric] for metric in item['metrics'].keys())
            
            # Assume 'train_accuracy' is your target
            actual_accuracy = item['test_accuracy']
            
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
    
    def get_ensemble_weight(self, model_result, best_params):
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

        # Calculate the ensemble weight as the sum of products of each metric's value and its best parameter.
        ensemble_weight = sum(model_result['metrics'][metric] * best_params[metric] for metric in model_result['metrics'])

        return ensemble_weight

    # add = lambda x, y: x + y
    # print(add(5, 3))  # Output: 8
    
    # zcp_preds = {}
    # for zcp_name in zc_proxies:
    #   zcp_test = [{'zero_cost_scores': eval_zcp(t_arch, zcp_name, train_loader)} for t_arch in tqdm(xtest)]
    #   zcp_pred = [s['zero_cost_scores'][zcp_name] for s in zcp_test]
    #   zcp_preds[zcp_name] = zcp_pred


    # ensemble_preds = []
    # for i in range(train_size):
    #   ensemble_preds.append(sum([zcp_preds[zcp_name][i] * best_params[zcp_name] for zcp_name in zc_proxies])) 

    # print(ensemble_preds)

    # ens_metrics = evaluate_predictions(ytest, ensemble_preds, plot=False,
    #                                 title=f"NB201 accuracies vs {zcp_name}")

    # print("ens_metrics:  ", ens_metrics)

    def search(self, search_space) -> optuna.study.Study:
        print("search_space:  ", search_space)

        # import pdb
        # pdb.set_trace()
        
        study_kwargs = {
            "sampler": self.sampler_map(self.config["setup"]["sampler"]),
            "direction": self.config["setup"]["direction"]
        }
        
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

        best_params = study.best_params
        print("best_params:  ", best_params)


        model_results = self.combined_list(search_space.zcp_results)

        for x in model_results:
            x['metrics']['ensemble'] = self.get_ensemble_weight(x, best_params)

            # import pdb
            # pdb.set_trace()
        
        print("model_results: ", model_results)

        self._save_study(study, self.save_dir / "study.pkl")
        self._save_search_dataframe(study, search_space, self.save_dir / "log.json")
        self._save_best_zero_cost(study, search_space, model_results, self.save_dir / "metrics.json")

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
    def _save_best_zero_cost(study, search_space, model_results, save_path):
        # import pdb
        # pdb.set_trace()

        result_dict = {}
        for item in search_space.zcp_results:
            key = list(item.keys())[0]
            values = item[key]
            result_dict[key] = {
                'test_spearman': values['test_spearman'], 
                'train_spearman': values['train_spearman'], 
                "zero_cost_weight": study.best_params[key]
            }

        print("result_dict:  ", result_dict)

        df = pd.DataFrame(
            columns=[
                "number",
                "result_dict",
                "model_results"
            ]
        )
        row = [
            0,
            result_dict,
            model_results,
            ]
        
        df.loc[len(df)] = row
        df.to_json(save_path, orient="index", indent=4)

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
