import gymnasium as gym
import copy
import torch
import numpy as np
import math
from pprint import pprint
import random
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from typing import Any, Dict, List, Tuple
from stable_baselines3.common.callbacks import BaseCallback
from abc import ABC, abstractmethod
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.utils import get_mase_op, get_node_actual_target
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
)

class BaseMixedPrecisionEnv(gym.Env, ABC):
    @abstractmethod
    def __init__(self, config, search_space, sw_runner, hw_runner, data_module, episode_max_len):
        pass

    @abstractmethod
    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state and return an initial observation."""
        pass

    @abstractmethod
    def step(self, action):
        """Take a step using the given action."""
        pass

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool = True):
        """Compute software metrics for the current model and config."""
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.sw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.sw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool = True):
        """Compute hardware metrics for the current model and config."""
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics


class MixedPrecisionEnv(BaseMixedPrecisionEnv):
    def __init__(self, config, search_space, sw_runner, hw_runner, data_module, episode_max_len):
        if search_space is None:
            raise ValueError("search_space cannot be None")
        
        self.config = config
        self.search_space = search_space
        self.sw_runner = sw_runner
        self.hw_runner = hw_runner
        self.data_module = data_module
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        self.directions = [self.config["metrics"][k]["direction"] for k in self.metric_names]
        
        self.direction_multipliers = {
            metric: (-1 if self.config["metrics"][metric]["direction"] == "maximize" else 1)
            for metric in self.metric_names
        }
        self._define_observation_space()
        self._define_action_space()
        self.cur_obs = None
        self.episode_len = 0
        self.episode_max_len = episode_max_len


    def _define_observation_space(self):
        """Defines the observation space based on the search space."""
        self.observation_space = Dict({
            "cost": Box(0.0, 10e4, shape=(1,)),
            "accuracy": Box(0.0, 1.0, shape=(1,)),
            "average_bitwidth": Box(2.0, 32.0, shape=(1,))
        })
        for key, choices in self.search_space.choices_flattened.items():
            self.observation_space[key] = Discrete(len(choices))

    def _define_action_space(self):
        num_options_per_key = [len(options) for options in self.search_space.choices_flattened.values()] 
        self.action_space = MultiDiscrete(num_options_per_key)
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        """Resets the episode and returns the initial observation of the new one."""
        if seed is None:
            seed = self.seed()
        else:
            self.seed(seed)

        self.episode_len = 0
        self.cur_obs = self.observation_space.sample()
        return self.cur_obs, {}
    
    def step(self, action):
        flattened_sampled_indexes = self._action_to_config(action)
        sampled_config = self.search_space.flattened_indexes_to_config(flattened_sampled_indexes)
        model = self.search_space.rebuild_model(sampled_config)
        software_metrics = self.compute_software_metrics(model, sampled_config)
        hardware_metrics = self.compute_hardware_metrics(model, sampled_config)
        metrics = software_metrics | hardware_metrics
        scaled_metrics = {}
        cost = 0
        for metric_name in self.metric_names:
            scaled_metric_value = self.config["metrics"][metric_name]["scale"] * metrics[metric_name] * self.direction_multipliers[metric_name]
            scaled_metrics[metric_name] = scaled_metric_value
            cost += scaled_metric_value
        self.cur_obs['accuracy'] = np.array([software_metrics['accuracy']], dtype=np.float32)
        self.cur_obs['average_bitwidth'] = np.array([hardware_metrics['average_bitwidth']], dtype=np.float32)
        self.cur_obs['cost'] = np.array([cost], dtype=np.float32)
        self.cur_obs.update(flattened_sampled_indexes)
        
        reward = -cost

        self.episode_len += 1
        done = self.episode_len >= self.episode_max_len
        truncated = self.episode_len >= self.episode_max_len
        
        info = {"reward": reward, "loss": software_metrics['loss'], "average_bitwidth": hardware_metrics['average_bitwidth'], "accuracy": software_metrics['accuracy'], "memory density": hardware_metrics['memory_density']}
        return self.cur_obs, reward, done, truncated, info
    

    def _action_to_config(self, action):
        config = {}
        choices_flattened = self.search_space.choices_flattened
        action_idx = 0
        for key, choices in choices_flattened.items():
            choice_idx = action[action_idx]
            config[key] = choice_idx
            action_idx += 1
        return config
    

class MixedPrecisionEnvHiLo(BaseMixedPrecisionEnv):
    def __init__(self, config, search_space, sw_runner, hw_runner, data_module, episode_max_len):
        if search_space is None:
            raise ValueError("search_space cannot be None")
        self.config = config
        self.search_space = search_space
        self.sw_runner = sw_runner
        self.hw_runner = hw_runner
        self.data_module = data_module
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        self.directions = [self.config["metrics"][k]["direction"] for k in self.metric_names]
        
        self.direction_multipliers = {
            metric: (-1 if self.config["metrics"][metric]["direction"] == "maximize" else 1)
            for metric in self.metric_names
        }
        self.directions = [self.config["metrics"][k]["direction"] for k in self.metric_names]
        self._define_observation_space()
        self._define_action_space()
        self.cur_obs = None
        self.episode_len = 0
        self.episode_max_len = episode_max_len
        self.model_config = {key: 0 for key in self.search_space.choices_flattened.keys()}

    def _define_observation_space(self):
        self.observation_space = Dict({
            "cost": Box(0.0, 10e4, shape=(1,)),
            "accuracy": Box(0.0, 1.0, shape=(1,)),
            "average_bitwidth": Box(2.0, 32.0, shape=(1,))
        })
        for key, choices in self.search_space.choices_flattened.items():
            self.observation_space[key] = Discrete(len(choices))

    def _define_action_space(self):
        num_actions = len(self.search_space.choices_flattened)
        # Each action can be -1, 0, or 1
        self.action_space = MultiDiscrete([3] * num_actions)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.seed()
        else:
            self.seed(seed)

        self.episode_len = 0
        self.cur_obs = self.observation_space.sample()
        self.model_config = {
        key: len(options) // 2 for key, options in self.search_space.choices_flattened.items()
        }
        return self.cur_obs, {}
    
    def step(self, action):
        self._action_to_config(action)
        sampled_config = self.search_space.flattened_indexes_to_config(self.model_config)
        model = self.search_space.rebuild_model(sampled_config)
    
        software_metrics = self.compute_software_metrics(model, sampled_config)
        hardware_metrics = self.compute_hardware_metrics(model, sampled_config)
        metrics = software_metrics | hardware_metrics
        scaled_metrics = {}
        cost = 0
        for metric_name in self.metric_names:
            scaled_metric_value = self.config["metrics"][metric_name]["scale"] * metrics[metric_name] * self.direction_multipliers[metric_name]
            scaled_metrics[metric_name] = scaled_metric_value
            cost += scaled_metric_value
        
        self.cur_obs['accuracy'] = np.array([software_metrics['accuracy']], dtype=np.float32)
        self.cur_obs['average_bitwidth'] = np.array([hardware_metrics['average_bitwidth']], dtype=np.float32)
        self.cur_obs['cost'] = np.array([cost], dtype=np.float32)
        self.cur_obs.update(self.model_config)
        
        reward = -cost
        self.episode_len += 1
        done = self.episode_len >= self.episode_max_len
        truncated = self.episode_len >= self.episode_max_len
        
        info = {"reward": reward, "loss": software_metrics['loss'], "average_bitwidth": hardware_metrics['average_bitwidth'], "accuracy": software_metrics['accuracy'], "memory density": hardware_metrics['memory_density']}
        #if done:
            #reward = reward*2
        return self.cur_obs, reward, done, truncated, info
    
    def _action_to_config(self, action):
        action_idx = 0
        for key, choices in self.search_space.choices_flattened.items():
            current_value = self.model_config[key]
            change = action[action_idx] - 1  # Map from [0, 1, 2] to [-1, 0, 1]
            new_value = max(min(current_value + change, len(choices) - 1), 0)
            self.model_config[key] = new_value
            action_idx += 1


class MixedPrecisionEnvHiLo(BaseMixedPrecisionEnv):
    def __init__(self, config, search_space, sw_runner, hw_runner, data_module, episode_max_len):
        if search_space is None:
            raise ValueError("search_space cannot be None")
        self.config = config
        self.search_space = search_space
        self.sw_runner = sw_runner
        self.hw_runner = hw_runner
        self.data_module = data_module
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        self.directions = [self.config["metrics"][k]["direction"] for k in self.metric_names]
        
        self.direction_multipliers = {
            metric: (-1 if self.config["metrics"][metric]["direction"] == "maximize" else 1)
            for metric in self.metric_names
        }
        self.directions = [self.config["metrics"][k]["direction"] for k in self.metric_names]
        self._define_observation_space()
        self._define_action_space()
        self.cur_obs = None
        self.episode_len = 0
        self.episode_max_len = episode_max_len
        self.model_config = {key: 0 for key in self.search_space.choices_flattened.keys()}

    def _define_observation_space(self):
        self.observation_space = Dict({
            "cost": Box(0.0, 10e4, shape=(1,)),
            "accuracy": Box(0.0, 1.0, shape=(1,)),
            "average_bitwidth": Box(2.0, 32.0, shape=(1,))
        })
        for key, choices in self.search_space.choices_flattened.items():
            self.observation_space[key] = Discrete(len(choices))

    def _define_action_space(self):
        num_actions = len(self.search_space.choices_flattened)
        # Each action can be -1, 0, or 1
        self.action_space = MultiDiscrete([3] * num_actions)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.seed()
        else:
            self.seed(seed)

        self.episode_len = 0
        self.cur_obs = self.observation_space.sample()

        self.model_config = {
        key: len(options) // 2 for key, options in self.search_space.choices_flattened.items()
    }
        return self.cur_obs, {}
    
    def step(self, action):
        # Update model configuration based on action
        self._action_to_config(action)
        sampled_config = self.search_space.flattened_indexes_to_config(self.model_config)
        model = self.search_space.rebuild_model(sampled_config)
    
        software_metrics = self.compute_software_metrics(model, sampled_config)
        hardware_metrics = self.compute_hardware_metrics(model, sampled_config)
        metrics = software_metrics | hardware_metrics
        scaled_metrics = {}
        cost = 0
        for metric_name in self.metric_names:
            # Apply scaling and direction multiplier from pre-computed values
            scaled_metric_value = self.config["metrics"][metric_name]["scale"] * metrics[metric_name] * self.direction_multipliers[metric_name]
            scaled_metrics[metric_name] = scaled_metric_value
            cost += scaled_metric_value
        
        self.cur_obs['accuracy'] = np.array([software_metrics['accuracy']], dtype=np.float32)
        self.cur_obs['average_bitwidth'] = np.array([hardware_metrics['average_bitwidth']], dtype=np.float32)
        self.cur_obs['cost'] = np.array([cost], dtype=np.float32)
        self.cur_obs.update(self.model_config)
        
        # Adjust reward calculation based on your scenario
        reward = -cost
        # Determine if the episode is done
        self.episode_len += 1
        done = self.episode_len >= self.episode_max_len
        truncated = self.episode_len >= self.episode_max_len
        
        info = {"reward": reward, "loss": software_metrics['loss'], "average_bitwidth": hardware_metrics['average_bitwidth'], "accuracy": software_metrics['accuracy'], "memory density": hardware_metrics['memory_density']}
        #if done:
            #reward = reward*2
        return self.cur_obs, reward, done, truncated, info
    
    def _action_to_config(self, action):
        action_idx = 0
        for key, choices in self.search_space.choices_flattened.items():
            current_value = self.model_config[key]
            change = action[action_idx] - 1  # Map from [0, 1, 2] to [-1, 0, 1]
            new_value = max(min(current_value + change, len(choices) - 1), 0)
            self.model_config[key] = new_value
            action_idx += 1


class MixedPrecisionPaper(BaseMixedPrecisionEnv):
    def __init__(self, config, search_space, sw_runner, hw_runner, data_module, episode_max_len):
        if search_space is None:
            raise ValueError("search_space cannot be None")
        self.search_space = search_space
        self.sw_runner = sw_runner
        self.hw_runner = hw_runner
        self.data_module = data_module
        self.config = config
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        self.best_performance = {}
        self.best_sample = {}
        self.metric_values = {}

        graph = MaseGraph(self.search_space.model)
        graph, _ = init_metadata_analysis_pass(graph)
        graph, _ = add_common_metadata_analysis_pass(
            graph, {"dummy_in": self.search_space.dummy_input}
        )
        layer_info = {}
        idx = 0
        for node in graph.fx_graph.nodes:
            if get_mase_op(node) == "linear":
                target = get_node_actual_target(node)
                layer_info[node.name] = [idx, target.in_features, target.out_features, 1, 0]
                idx += 1
            elif get_mase_op(node) == "conv2d":
                target = get_node_actual_target(node)
                layer_info[node.name] = [
                    idx, target.in_channels, target.out_channels, target.kernel_size[0], target.stride[0]]
                idx += 1
            else:
                target = get_node_actual_target(node)
                layer_info[node.name] = [idx, 0, 0, 0, 0]
                idx += 1

        self.obs_list = []
        self.action_list = []
        self.layer_names = []
        self.sample = {}

        for name, choices in self.search_space.choices_flattened.items():
            if len(choices) == 1:
                self.sample[name] = 0
                continue
            self.layer_names.append(name)
            _name = name.split("/")
            cur_obs = layer_info[_name[0]].copy()
            if _name[2] == "data_in_width":
                cur_obs.append(1)
            elif _name[2] == "weight_width":
                cur_obs.append(2)
            elif _name[2] == "bias_width":
                cur_obs.append(3)
            else:
                cur_obs.append(0)
            self.obs_list.append(cur_obs)
            self.action_list.append(sorted(choices))

        self.episode_len = 0
        self.obs_list = np.array(self.obs_list)

        lower_bound = np.min(self.obs_list, axis=0)
        upper_bound = np.max(self.obs_list, axis=0)
        self.observation_space = Box(
            low=np.append(lower_bound, min([min(sub) for sub in self.action_list])),
            high=np.append(upper_bound, max([max(sub) for sub in self.action_list])),
        )
        self.action_space = Box(low=0, high=1.0)

    def eval(self, sampled_indexes):
        sampled_config = self.search_space.flattened_indexes_to_config(sampled_indexes)
        model = self.search_space.rebuild_model(sampled_config)

        software_metrics = self.compute_software_metrics(model, sampled_config)
        hardware_metrics = self.compute_hardware_metrics(model, sampled_config)

        metrics = software_metrics | hardware_metrics

        scaled_metrics = {}
        for metric_name in self.metric_names:
            lower_bound = self.config["metrics"][metric_name].get("lower_bound", 0)
            upper_bound = self.config["metrics"][metric_name].get("upper_bound", 1)
            direction = self.config["metrics"][metric_name].get("direction", "maximize")
            if direction == "maximize":
                unit_metric = max(
                    max(lower_bound, metrics[metric_name]) - lower_bound, 0
                ) / (upper_bound - lower_bound)
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
            self.layers, self.layer_types = self.get_layers_of_graph(model)
            print(f"new highest reward: {reward:.4f}")
            for metric_name in self.metric_names:
                print(f"{metric_name}: {metrics[metric_name]:.4f}")
                self.metric_values[metric_name] = metrics[metric_name]
        return reward, metrics
    
    def get_layers_of_graph(self, graph):
        layers = []
        layer_types = []
        for node in graph.fx_graph.nodes:
            if node.meta["mase"].module is not None:
                layers.append(str(node))
                layer_types.append(type(node.meta["mase"].module).__name__)
        return layers, layer_types

    def reset(self, *, seed=None, options=None):
        self.episode_len = 0
        obs = np.append(
            self.obs_list[self.episode_len, :], min(self.action_list[self.episode_len])
        ).astype(np.float32)
        return obs, {}

    def step(self, action):
        choices = self.action_list[self.episode_len]
        action = int(action * len(choices) - 1e-2)
        self.sample[self.layer_names[self.episode_len]] = action
        reward = 0
        terminated = truncated = False
        self.episode_len += 1
        info = {}
        if self.episode_len == len(self.obs_list):
            self.episode_len = 0
            terminated = truncated = True
            reward, metrics = self.eval(self.sample)
            info = {"reward": reward, "average_bitwidth": metrics['average_bitwidth'], "accuracy": metrics['accuracy']}
        obs = self.obs_list[self.episode_len].copy()
        obs = np.append(obs, choices[action]).astype(np.float32)
        return obs, reward, terminated, truncated, info


#################### Register the environments ####################

Env_id = 'RL/MixedPrecisionEnv-v0'
gym.envs.registration.register(
    id=Env_id,
    entry_point=MixedPrecisionEnv,
    max_episode_steps=10,
    reward_threshold=500
)

Env_id_Hi_Lo = 'RL/MixedPrecisionEnvHiLo-v0'
gym.envs.registration.register(
    id=Env_id_Hi_Lo,
    entry_point=MixedPrecisionEnvHiLo,
    max_episode_steps=10,
    reward_threshold=500
)

Env_id_Hi_Lo = 'RL/MixedPrecisionPaper-v0'
gym.envs.registration.register(
    id=Env_id_Hi_Lo,
    entry_point=MixedPrecisionPaper,
    max_episode_steps=100000,
    reward_threshold=500
)

env_map = {
    'mixed_precision': MixedPrecisionEnv,
    'mixed_precision_hi_lo': MixedPrecisionEnvHiLo,
    'mixed_precision_paper': MixedPrecisionPaper,
}

registered_env_map = {
    'mixed_precision': 'RL/MixedPrecisionEnv-v0',
    'mixed_precision_hi_lo': "RL/MixedPrecisionEnvHiLo-v0",
    'mixed_precision_paper': "RL/MixedPrecisionPaper-v0",
}