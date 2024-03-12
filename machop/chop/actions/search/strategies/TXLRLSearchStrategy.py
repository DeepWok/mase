import torch
import logging

from chop.actions.search.strategies.base import SearchStrategyBase
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from .TXLenv import MixedPrecisionEnv

logger = logging.getLogger(__name__)

algorithm_map = {
    "ppo": PPO,
    "a2c": A2C,
}
env_map = {
    "MixedPrecisionEnv": MixedPrecisionEnv
}
class SearchStrategyRL(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        setup = self.config['setup']
        self.device = setup.get('device', 'cpu')
        self.total_trials = setup["total_trials"]
        algorithm_name = setup.get('algorithm', 'ppo')
        env_name = setup.get('env', 'MixedPrecisionEnv')
        if algorithm_name not in algorithm_map:
            raise ValueError(f"Unsupported algorithm name: {algorithm_name}")
        if env_name not in env_map:
            raise ValueError(f"Unsupported env name: {env_name}")
        self.algorithm = algorithm_map[algorithm_name]
        self.env = env_map[env_name]
        self.search_space = None
        self.best_performance = 0
        self.best_sample = {}

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
            scaled_metrics[metric_name] = (
                    self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
            )
        if sum(scaled_metrics.values()) > self.best_performance:
            self.best_performance = sum(scaled_metrics.values())
            self.best_sample = sampled_config
            print(f'highest reward: {sum(scaled_metrics.values()):.4f}')
            for metric_name in self.metric_names:
                print(f'{metric_name}: {metrics[metric_name]:.4f}')
        return sum(scaled_metrics.values())

    def search(self, search_space):
        self.search_space = search_space
        env = self.env(config={"search_space": self.search_space, "run_trial": self.run_trial})
        check_env(env)
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=f"{self.save_dir}/logs/")
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

        # TODO report best performed sample
