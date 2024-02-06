import optuna
import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib
import numpy as np
import time

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
        return {"sw_metrics":software_metrics,"hw_metrics":hardware_metrics,"scaled_metrics": scaled_metrics, "aggregate":scaled_metrics_aggregrate} 
        
    def search(self, search_space) -> optuna.study.Study:
        start_time = time.time()
        keys = search_space.choice_lengths_flattened.keys()
        keys_list = list(keys)

        variable_ranges = []
        no_combinations = 1
        for k,v in search_space.choice_lengths_flattened.items():
            variable_ranges.append(list(range(v)))
            no_combinations = no_combinations*v

        all_combinations = brute_force_combinations(variable_ranges)
        results={}
        sw_metrics=[]
        hw_metrics=[]
        scaled_metrics=[]
        scaled_metrics_aggregrate=[]
        all_metrics = []
        for i in range(no_combinations):
            sampled_index = lists_to_dict(keys_list, all_combinations[i])
            results = self.objective(sampled_index, search_space)
            sw_metrics.append(results["sw_metrics"])
            hw_metrics.append(results["hw_metrics"])
            scaled_metrics.append(results["scaled_metrics"]) 
            scaled_metrics_aggregrate.append(results["aggregate"]) 

            metrics_for_combination = {
              "Config Number": i,
              "Software Metrics": results["sw_metrics"],
              "Hardware Metrics": results["hw_metrics"],
              "Scaled Metrics": results["scaled_metrics"],
            }
            all_metrics.append(metrics_for_combination)
            self.visualizer.log_metrics(metrics=scaled_metrics[i], step=i)
        best_index = np.argmax(scaled_metrics_aggregrate)

        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Total time taken for search: %f seconds", total_time)

        df = pd.DataFrame(all_metrics)
        print("Results for all Configurations in Search Space")
        print(tabulate(df, headers='keys', tablefmt='psql'))

        
        best_sw_metrics = sw_metrics[best_index] 
        best_hw_metrics = hw_metrics[best_index] 
        best_scaled_metric = scaled_metrics[best_index]  
        best_config = search_space.flattened_indexes_to_config(lists_to_dict(keys_list, all_combinations[best_index]))
        search_result = {"index":best_index,"sw_metrics":best_sw_metrics,"hw_metrics":best_hw_metrics,"scaled_metrics":best_scaled_metric,"config": best_config}
       
        self._save_best(search_result, self.save_dir / "study.pkl")
        print("Best Configuration is :", best_config)
        return search_result

    @staticmethod
    def _save_best(search_result, save_path):
        df = pd.DataFrame(
            columns=[
                "Config number",
                "software_metrics",
                "hardware_metrics",
                "scaled_metrics",
                "sampled_config",
            ]
        )
        row = [
                search_result["index"],
                search_result["sw_metrics"],
                search_result["hw_metrics"],
                search_result["scaled_metrics"],
                search_result["config"],
            ]    
        df.loc[len(df)] = row
       
        df.to_json(save_path, orient="index", indent=4)

        txt = "Best Result(s):\n"
        df_truncated = df.loc[
            :, ["Config number", "software_metrics", "hardware_metrics", "scaled_metrics"]
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
        logger.info(f"Best Result(s):\n{txt}")
        return df
