import gymnasium as gym
import copy
import torch
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from stable_baselines3.common.callbacks import BaseCallback


################################################ LEGACY CODE - NOT FOR USE ##############################################
class LLMMixedPrecisionEnv(gym.Env): 
    def __init__(self, config):
        # Make the space (for actions and observations) configurable.
        # Since actions should repeat observations, their spaces must be the
        # same.
        search_space = config.get("search_space", None)
        runner = config.get("runner", None)
        self.search_space, self.runner = search_space, runner
        if search_space is None:
            raise ValueError(f"search_space cannot be None")

        indexes, self.config_seed = search_space.build_opt_seed_and_indexes()

        # observation space definition
        self.observation_space = Dict({})
        for k, v in search_space.index_ranges.items():
            self.observation_space[k] = Discrete(v)
        per_layer_space = search_space.per_layer_search_space
        flattend_per_layer_space = search_space.transform_nested_dict_to_flat_dict(
            per_layer_space
        )
        my_list = list(flattend_per_layer_space.values())

        # action space definition
        self.action_space = MultiDiscrete((search_space.get_num_layers(), *my_list))

        self.cur_obs = None
        self.episode_len = 0

    def sample(self, indexes, quant_config_seed, search_space):
        sampled_config = search_space.config_sampler(
            indexes=indexes, config_seed=quant_config_seed
        )
        sampled_config = search_space.config_parser(
            sampled_config, search_space.get_num_layers()
        )
        model = search_space.rebuild_model(sampled_config)
        return self.compute_software_metric(model)

    def build_sample_based_on_action(self, action, indexes):
        layer_id = action[0]
        layer_config = action[1:]

        layer_index = copy.deepcopy(indexes[f"model_layer_{layer_id}"])
        flat_layer = self.search_space.transform_nested_dict_to_flat_dict(layer_index)
        i = 0
        for k, v in flat_layer.items():
            flat_layer[k] = layer_config[i]
            i += 1
        nested_layer = self.search_space.transform_flat_dict_to_nested_dict(flat_layer)
        indexes[f"model_layer_{layer_id}"] = nested_layer
        return indexes

    def compute_software_metric(self, model):
        loss, _ = self.runner(model)
        cost = 10 # TODO: fix
        return loss + cost["avg_bit"]

    def reset(self, *, seed=None, options=None):
        """Resets the episode and returns the initial observation of the new one."""
        # Reset the episode len.
        self.episode_len = 0
        # Sample a random number from our observation space.
        self.cur_obs = self.observation_space.sample()
        # Return initial observation.
        return self.cur_obs, {}

    def step(self, action):
        """Takes a single step in the episode given `action`

        Returns:
            New observation, reward, done-flag, info-dict (empty).
        """
        # loss = self.cur_obs.pop('loss', None)
        cur_obs = self.cur_obs
        obs = self.search_space.transform_flat_dict_to_nested_dict(cur_obs)
        indexes = self.build_sample_based_on_action(action, obs)
        metric = self.sample(indexes, self.config_seed, self.search_space)
        metric = metric.cpu().detach().numpy()

        # Set `truncated` flag after 10 steps.
        self.episode_len += 1
        terminated = False
        truncated = self.episode_len >= 10

        # TODO: this reward needs redesign, apparently
        reward = -metric
        indexes_flat = self.search_space.transform_nested_dict_to_flat_dict(indexes)
        # indexes_flat['loss'] = metric
        return indexes_flat, reward, terminated, truncated, {"loss": metric}

##############################################################################################################################



# TODO:
# 1 - Discuss the following design decisions:
# MixedPrecisionEnv:
# - Does random sampling of the search space for the initial observation affect perf?
# - Does the reward function affect perf?
      
# MixedPrecisionEnvHiLo:
# - Does random sampling of the search space for the initial observation affect perf? (0 vs median)
# - Does the number of steps affect perf?
# - Does the reward function affect perf?
    
# Meeting:
# - Plot accuracy for same bitwidth
    

class MixedPrecisionEnv(gym.Env):
    def __init__(self, config, search_space, sw_runner, hw_runner, data_module):
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
        self._define_observation_space()
        self._define_action_space()
        self.cur_obs = None
        self.episode_len = 0

    def _define_observation_space(self):
        """Defines the observation space based on the search space."""
        self.observation_space = Dict({"cost": Box(0.0, 10e4, shape=(1,))})
        for key, choices in self.search_space.choices_flattened.items():
            self.observation_space[key] = Discrete(len(choices))

    def _define_action_space(self):
        """Defines the action space based on the search space."""
        self.action_space = MultiDiscrete(self.search_space.get_action_space_options())
    
    def reset(self, *, seed=None, options=None):
        """Resets the episode and returns the initial observation of the new one."""
        self.episode_len = 0
        self.cur_obs = self.observation_space.sample()
        return self.cur_obs, {}

    def step(self, action):
        """
        Apply the mixed-precision configuration defined by `action`, compute the model's metrics,
        update the environment's state, calculate the reward, and check if the episode should end.
        """
        # Map action to model configuration changes
        flattened_sampled_indexes = self._action_to_config(action)
        sampled_config = self.search_space.flattened_indexes_to_config(flattened_sampled_indexes)
        model = self.search_space.rebuild_model(sampled_config)
    
        software_metrics = self.compute_software_metrics(model, sampled_config)
        hardware_metrics = self.compute_hardware_metrics(model, sampled_config)

        metrics = software_metrics | hardware_metrics
        scaled_metrics = {}
        for metric_name in self.metric_names:
            scaled_metrics[metric_name] = (
                self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
            )

        cost = sum(scaled_metrics.values())
        self.cur_obs['cost'] = np.array([cost], dtype=np.float32)
        self.cur_obs.update(sampled_config)
        reward = -cost

        # Determine if the episode is done
        self.episode_len += 1
        done = self.episode_len >= 10 #TODO: not sure what should this be used for
        truncated = self.episode_len >= 10 #TODO: not sure what should this be used for
    
        # Optional additional info about the step
        info = {"loss": software_metrics['loss'], "average_bitwidth": hardware_metrics['average_bitwidth']}

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

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool = True): # TODO: these functions should be moved to the runner
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

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool = True):
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics



class MixedPrecisionEnvHiLo(gym.Env):
    def __init__(self, config, search_space, sw_runner, hw_runner, data_module):
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
        self._define_observation_space()
        self._define_action_space()
        self.cur_obs = None
        self.episode_len = 0
        # Initialize model configuration with the lowest precision settings
        self.model_config = {key: 0 for key in self.search_space.choices_flattened.keys()}

    def _define_observation_space(self):
        self.observation_space = Dict({"cost": Box(0.0, 10e4, shape=(1,))})
        for key, choices in self.search_space.choices_flattened.items():
            self.observation_space[key] = Discrete(len(choices))

    def _define_action_space(self):
        num_actions = len(self.search_space.choices_flattened)
        # Each action can be -1, 0, or 1
        self.action_space = MultiDiscrete([3] * num_actions)
    
    def reset(self, *, seed=None, options=None):
        self.episode_len = 0
        self.cur_obs = self.observation_space.sample()
        # Reset model configuration to the lowest precision settings
        self.model_config = {key: 0 for key in self.search_space.choices_flattened.keys()}

        # Another option is init on median values: 
        # Initialize model configuration to the median precision setting for each parameter
        #self.model_config = {
        #    key: choices[math.floor(len(choices) / 2)] for key, choices in self.search_space.choices_flattened.items()
        #}

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
        for metric_name in self.metric_names:
            scaled_metrics[metric_name] = (
                self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
            )

        cost = sum(scaled_metrics.values())
        self.cur_obs['cost'] = np.array([cost], dtype=np.float32)
        self.cur_obs.update(sampled_config)
        reward = -cost

        self.episode_len += 1
        done = self.episode_len >= 10
        truncated = self.episode_len >= 10
    
        info = {"loss": software_metrics['loss'], "average_bitwidth": hardware_metrics['average_bitwidth']}

        return self.cur_obs, reward, done, truncated, info

    def _action_to_config(self, action):
        action_idx = 0
        for key, choices in self.search_space.choices_flattened.items():
            current_value = self.model_config[key]
            change = action[action_idx] - 1  # Map from [0, 1, 2] to [-1, 0, 1]
            new_value = max(min(current_value + change, len(choices) - 1), 0)
            self.model_config[key] = new_value
            action_idx += 1

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool = True): # TODO: these functions should be moved to the runner
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

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool = True):
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics


env_map = {
    "mixed_precision": MixedPrecisionEnv,
    'mixed_precision_hi_lo': MixedPrecisionEnvHiLo,
    #"llm_mixed_precision": LLMMixedPrecisionEnv,
}