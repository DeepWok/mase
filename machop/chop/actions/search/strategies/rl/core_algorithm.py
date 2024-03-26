import torch
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import datetime
import json

from chop.actions.search.strategies.base import SearchStrategyBase
from stable_baselines3 import A2C, PPO, DDPG, SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from .env import MixedPrecisionEnv

logger = logging.getLogger(__name__)

algorithm_map = {
    "ppo": PPO,
    "a2c": A2C,
    "ddpg": DDPG,
    "sac": SAC,
}
env_map = {"MixedPrecisionEnv": MixedPrecisionEnv}

class SearchStrategyRL(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        setup = self.config["setup"]
        self.device = setup.get("device", "cpu")
        self.total_trials = setup["total_trials"]
        algorithm_name = setup.get("algorithm", "a2c")
        env_name = setup.get("env", "MixedPrecisionEnv")
        if algorithm_name not in algorithm_map:
            raise ValueError(f"Unsupported algorithm name: {algorithm_name}")
        if env_name not in env_map:
            raise ValueError(f"Unsupported env name: {env_name}")
        self.algorithm = algorithm_map[algorithm_name]
        self.env = env_map[env_name]
        self.search_space = None
        self.best_performance = {}
        self.best_sample = {}
        self.metric_values = {}

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

    def run_trial(self, sampled_indexes):
        """
        compute metrics of a sample in search space
        """
        # parse the sample
        sampled_config = self.search_space.flattened_indexes_to_config(sampled_indexes)

        # rebuild model with sampled configuration
        is_eval_mode = self.config.get("eval_mode", True)
        model = self.search_space.rebuild_model(sampled_config, is_eval_mode)

        # get metrics
        software_metrics = self.compute_software_metrics(
            model, sampled_config, is_eval_mode
        )
        hardware_metrics = self.compute_hardware_metrics(
            model, sampled_config, is_eval_mode
        )
        metrics = software_metrics | hardware_metrics

        # sum the metrics with configured scales
        scaled_metrics = {}
        for metric_name in self.metric_names:
            # default range is [0,1]
            upper_bound = self.config["metrics"][metric_name].get("upper_bound", 1)
            lower_bound = self.config["metrics"][metric_name].get("lower_bound", 0)
            direction = self.config["metrics"][metric_name].get("direction", "maximize")
            if direction == "maximize":
                unit_metric = (max(
                    max(lower_bound, metrics[metric_name]) - lower_bound, 0
                ) / (upper_bound - lower_bound))
            else:
                unit_metric = max(
                    upper_bound - max(lower_bound, metrics[metric_name]), 0
                ) / (upper_bound - lower_bound)
            scaled_metrics[metric_name] = (
                    unit_metric * self.config["metrics"][metric_name]["scale"]
            )
        reward = sum(scaled_metrics.values())

        if reward > self.best_performance.get("reward", 0):
            self.best_performance["metrics"] = metrics
            self.best_performance["reward"] = reward
            self.best_sample = sampled_config
            self.layers, self.layer_types = get_layers_of_graph(model)
            logger.info(f"new highest reward: {reward:.4f}")
            for metric_name in self.metric_names:
                logger.info(f"{metric_name}: {metrics[metric_name]:.4f}")
                self.metric_values[metric_name] = metrics[metric_name]

        return reward

    def search(self, search_space):
        self.search_space = search_space
        env = self.env(
            config={"search_space": self.search_space, "run_trial": self.run_trial}
        )
        check_env(env)
        checkpoint_callback = CheckpointCallback(
            save_freq=1000, save_path=f"{self.save_dir}/logs/"
        )
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"{self.save_dir}/logs/best_model",
            log_path=f"{self.save_dir}/logs/results",
            eval_freq=500,
        )
        callback = CallbackList([checkpoint_callback, eval_callback])

        model = self.algorithm(
            "MlpPolicy",
            env,
            verbose=1,
            device=self.device,
            tensorboard_log=f"{self.save_dir}/logs/",
        )

        model.learn(
            total_timesteps=int(self.total_trials * len(env.obs_list)),
            progress_bar=True,
            callback=callback,
        )

        # save the result
        with open(f"{self.save_dir}/best_sample.json", 'w') as f:
            json.dump(
                {"performance": self.best_performance, "config": self.best_sample},
                f,
                indent=1
            )
        plot_config(
            self.best_sample,
            self.layers,
            self.layer_types,
            self.save_dir,
            self.metric_values,
        )


def get_layers_of_graph(graph):
    layers = []
    layer_types = []
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].module is not None:
            layers.append(str(node))
            layer_types.append(type(node.meta["mase"].module).__name__)
    return layers, layer_types


def plot_config(config, layers, layer_types, save_dir, metric_values):
    layer_dict = dict(zip(layers, layer_types))
    data_in_width = []
    weight_width = []
    bias_width = []
    layers_to_plot = []
    for layer in layers:
        if layer in config and config[layer]["config"]["name"] == "integer":
            values = config[layer]["config"]
            data_in_width.append(values["data_in_width"])
            weight_width.append(values["weight_width"])
            bias_width.append(values["bias_width"])
            layers_to_plot.append(layer_dict[layer])

    min_value = min([min(data_in_width), min(weight_width), min(bias_width)])
    max_value = max([max(data_in_width), max(weight_width), max(bias_width)])
    # title_fontsize = 16
    label_fontsize = 14
    ticks_fontsize = 14

    fig, ax = plt.subplots(figsize=(10, 4))

    bar_width = 0.2
    # Derive the x coordinates for each bar
    r1 = range(len(layers_to_plot))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Plot the bar chart
    rects1 = ax.bar(
        r1,
        data_in_width,
        color="skyblue",
        width=bar_width,
        edgecolor="white",
        label="Data In Width",
    )
    rects2 = ax.bar(
        r2,
        weight_width,
        color="lightgreen",
        width=bar_width,
        edgecolor="white",
        label="Weight Width",
    )
    rects3 = ax.bar(
        r3,
        bias_width,
        color="salmon",
        width=bar_width,
        edgecolor="white",
        label="Bias Width",
    )

    ax.set_ylabel("#bit", fontsize=label_fontsize)
    ax.set_xticks([r + bar_width for r in range(len(layers_to_plot))])
    ax.set_xticklabels(layers_to_plot, rotation=45, ha="right", fontsize=ticks_fontsize)
    ax.legend(fontsize=14, framealpha=0.3)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim([min_value - 0.5, max_value + 0.5])
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5)

    title_str = ", ".join(
        [f"{key}: {value:.4f}" for key, value in metric_values.items()]
    )
    plt.title(f"Bitwidth for layers ({title_str})", fontsize=16)
    plt.tight_layout()
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"{save_dir}/quantized_bit_width_{datetime_str}.png", dpi=300)
