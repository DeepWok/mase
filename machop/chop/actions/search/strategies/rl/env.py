import gymnasium as gym
import numpy as np
import torch
import random

from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

# from ray.rllib.algorithms.ppo import PPOConfig
from stable_baselines3.common.callbacks import BaseCallback


# from chop.passes.graph.analysis.total_bits_estimator import (
#     total_bits_module_analysis_pass,
# )


# class LLMMixedPrecisionEnv(gym.Env):
#     def __init__(self, config):
#         # Make the space (for actions and observations) configurable.
#         # Since actions should repeat observations, their spaces must be the
#         # same.
#         search_space = config.get("search_space", None)
#         runner = config.get("runner", None)
#         self.search_space, self.runner = search_space, runner
#         if search_space is None:
#             raise ValueError(f"search_space cannot be None")
#
#         indexes, self.config_seed = search_space.build_opt_seed_and_indexes()
#
#         # observation space definition
#         self.observation_space = Dict({})
#         for k, v in search_space.index_ranges.items():
#             self.observation_space[k] = Discrete(v)
#         per_layer_space = search_space.per_layer_search_space
#         flattend_per_layer_space = search_space.transform_nested_dict_to_flat_dict(
#             per_layer_space
#         )
#         my_list = list(flattend_per_layer_space.values())
#
#         # action space definition
#         self.action_space = MultiDiscrete((search_space.get_num_layers(), *my_list))
#
#         self.cur_obs = None
#         self.episode_len = 0
#
#     def sample(self, indexes, quant_config_seed, search_space):
#         sampled_config = search_space.config_sampler(
#             indexes=indexes, config_seed=quant_config_seed
#         )
#         sampled_config = search_space.config_parser(
#             sampled_config, search_space.get_num_layers()
#         )
#         model = search_space.rebuild_model(sampled_config)
#         return self.compute_software_metric(model)
#
#     def build_sample_based_on_action(self, action, indexes):
#         layer_id = action[0]
#         layer_config = action[1:]
#
#         layer_index = copy.deepcopy(indexes[f"model_layer_{layer_id}"])
#         flat_layer = self.search_space.transform_nested_dict_to_flat_dict(layer_index)
#         i = 0
#         for k, v in flat_layer.items():
#             flat_layer[k] = layer_config[i]
#             i += 1
#         nested_layer = self.search_space.transform_flat_dict_to_nested_dict(flat_layer)
#         indexes[f"model_layer_{layer_id}"] = nested_layer
#         return indexes
#
#     def compute_software_metric(self, model):
#         loss, _ = self.runner(model)
#         cost = 0
#         return loss + cost["avg_bit"]
#
#     def reset(self, *, seed=None, options=None):
#         """Resets the episode and returns the initial observation of the new one."""
#         # Reset the episode len.
#         self.episode_len = 0
#         # Sample a random number from our observation space.
#         self.cur_obs = self.observation_space.sample()
#         # Return initial observation.
#         return self.cur_obs, {}
#
#     def step(self, action):
#         """Takes a single step in the episode given `action`
#
#         Returns:
#             New observation, reward, done-flag, info-dict (empty).
#         """
#         # loss = self.cur_obs.pop('loss', None)
#         cur_obs = self.cur_obs
#         obs = self.search_space.transform_flat_dict_to_nested_dict(cur_obs)
#         indexes = self.build_sample_based_on_action(action, obs)
#         metric = self.sample(indexes, self.config_seed, self.search_space)
#         metric = metric.cpu().detach().numpy()
#
#         # Set `truncated` flag after 10 steps.
#         self.episode_len += 1
#         terminated = False
#         truncated = self.episode_len >= 10
#
#
#         reward = -metric
#         indexes_flat = self.search_space.transform_nested_dict_to_flat_dict(indexes)
#         # indexes_flat['loss'] = metric
#         return indexes_flat, reward, terminated, truncated, {"loss": metric}


class MixedPrecisionEnv(gym.Env):
    def __init__(self, config,verbose):
        # Make the space (for actions and observations) configurable.
        # Since actions should repeat observations, their spaces must be the
        # same.
        self.is_eval_mode = config.get("eval_mode", True)
        search_space = config.get("search_space", None)
        self.sum_scaled_metrics = config["sum_scaled_metrics"]
        self.sw_runner = config["sw_runner"]
        self.hw_runner = config["hw_runner"]
        self.metrics = config["metrics"]
        self.metric_names = list(sorted(self.metrics.keys()))
        self.data_module = config["data_module"]
        self.verbose = verbose

        # self.search_space, self.runner = search_space, runner
        self.search_space = search_space
        if search_space is None:
            raise ValueError(f"search_space cannot be None")

        # observation space definition
        self.observation_space = Dict(
            {"reward": Box(0.0, 10e4, shape=(1,), dtype=np.float32)}
        )
        # for k, v in search_space.search_space_flattened.items():
        action_space_list = []
        for k, v in search_space.choice_lengths_flattened.items():
            # k1, k2 = k.split(".")
            # new_k = "_and_".join([k1, k2])
            new_k = k
            self.observation_space[new_k] = Discrete(v)
            if v > 1:
                action_space_list.append(v)

        # action space definition
        # my_list = list(search_space.per_layer_search_space.values())
        # self.action_space = MultiDiscrete((search_space.num_layers, *my_list))
        self.action_space = MultiDiscrete(action_space_list)
        self.cur_obs = None
        self.episode = 0
        self.result = np.zeros(2)

    # def get_performance(self, sample):
    #     return self.runner.get_performance(
    #         sample, self.search_space, self.runner.runner
    #     )

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

    def objective(self, current_action):
        search_space = self.search_space
        sampled_indexes = {}
        new_observations = self.cur_obs
        if hasattr(search_space, "optuna_sampler"):
            sampled_config = {}
            print("Error: hasattr")
            pass
        else:
            j = 0
            for name, length in search_space.choice_lengths_flattened.items():
                if length > 1:
                    new_observations[name] = current_action[j]
                    sampled_indexes[name] = current_action[j]
                    # sampled_indexes[name]=new_observations[name]= 0
                    j = j + 1
                else:
                    new_observations[name] = np.int64(0)
                    sampled_indexes[name] = np.int64(0)
            sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)

        # is_eval_mode = self.config.get("eval_mode", True)
        model = search_space.rebuild_model(sampled_config, self.is_eval_mode)

        software_metrics = self.compute_software_metrics(
            model, sampled_config, self.is_eval_mode
        )
        hardware_metrics = self.compute_hardware_metrics(
            model, sampled_config, self.is_eval_mode
        )
        metrics = software_metrics | hardware_metrics
        scaled_metrics = {}
        for metric_name in self.metric_names:
            scaled_metrics[metric_name] = (
                self.metrics[metric_name]["scale"] * metrics[metric_name]
            )

        # self.visualizer.log_metrics(metrics=scaled_metrics, step=trial.number)

        if not self.sum_scaled_metrics:
            return list(scaled_metrics.values()), new_observations
        else:
            return sum(scaled_metrics.values()), new_observations

    def reset(self, *, seed=None, options=None):
        """Resets the episode and returns the initial observation of the new one."""
        # Sample a random number from our observation space.
        self.cur_obs = self.observation_space.sample()
        # Return initial observation.
        return self.cur_obs, {}

    def step(self, action):
        """Takes a single step in the episode given `action`

        Returns:
            New observation, reward, done-flag, info-dict (empty).
        """
        rwd = self.cur_obs["reward"]
        # get performance
        metrics, new_obs = self.objective(action)

        # reward here is positive, the smaller the better,
        # since return -reward in the end
        reward = 100 * (1 - metrics[0]) + (metrics[1])  # + 50
        # if metrics[0] > 0.5:
        #     reward += -50
        # Set a new observation (random sample).
        new_obs["reward"] = np.array([metrics[0]]).reshape((1,))
        # new_obs["metrics"] = metrics
        # for k, v in sample_idx.items():
        #     for k2, v2 in v.items():
        #         rebuild_sample[f"{k}_and_{k2}"] = v2
        self.cur_obs = new_obs

        # Set `truncated` flag after 10 steps.
        self.episode += 1
        terminated = False
        if rwd < reward:
            self.result[0] = metrics[0]
            self.result[1] = metrics[1]
        if self.episode % 20 == 0:
            truncated = True
            if self.verbose==1:
                print(f"Step: {self.episode}, reward: {reward}, accuracy: {metrics[0]}, average_bit: {metrics[1]}")

        else:
            truncated = False
            # print(f"Episode: {self.episode}, reward: {reward}, accuracy: {metrics[0]}, average_bit: {metrics[1]}")

        return self.cur_obs, -reward, terminated, truncated, {"loss": metrics}


env_map = {
    "mixed_precision": MixedPrecisionEnv,
    # "llm_mixed_precision": LLMMixedPrecisionEnv,
}
