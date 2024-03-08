import gymnasium as gym
import copy
import torch
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from stable_baselines3.common.callbacks import BaseCallback


########     Not used        ########
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

############################################################################################################



# TODO: 
#- add the hardware metrics and hardware runner
#- calculate the reward based a weighted sum of hardware metrics and software metrics
#- Perhaps, instead of setting the action to the precision directly, we can set the action to the precision change: 
# (eg. 0: down, 1: same, 2: up)
    

class MixedPrecisionEnv(gym.Env):
    def __init__(self, config, search_space, sw_runner, data_module):
        if search_space is None:
            raise ValueError("search_space cannot be None")
        
        self.search_space = search_space
        self.sw_runner = sw_runner
        self.data_module = data_module
        self._define_observation_space()
        self._define_action_space()
        self.cur_obs = None
        self.episode_len = 0

    def _define_observation_space(self):
        """Defines the observation space based on the search space."""
        self.observation_space = Dict({"loss": Box(0.0, 10e4, shape=(1,))})
        for key, choices in self.search_space.choices_flattened.items():
            self.observation_space[key] = Discrete(len(choices))

    def _define_action_space(self):
        """Defines the action space based on the search space."""
        self.action_space = MultiDiscrete(self.search_space.get_action_space_options())
    
    def reset(self, *, seed=None, options=None):
        """Resets the episode and returns the initial observation of the new one."""
        self.episode_len = 0
        self.cur_obs = self.observation_space.sample()
        # Return initial observation.
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
    
        # Compute the model's metrics, including the loss, for the current configuration
        software_metrics = self.compute_software_metrics(
            model, sampled_config
        )

        loss = software_metrics['loss']
        self.cur_obs['loss'] = np.array([loss], dtype=np.float32)
        self.cur_obs.update(flattened_sampled_indexes)
        reward = -loss

        # Determine if the episode is done
        self.episode_len += 1
        done = self.episode_len >= 10 #TODO: not sure
        truncated = self.episode_len >= 10 #TODO: not sure
    
        # Optional: additional info about this step
        info = {"loss": loss}

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

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool = True):
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

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool = True): # Not used yet
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
    #"llm_mixed_precision": LLMMixedPrecisionEnv,
}