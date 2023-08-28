import gymnasium as gym
import copy

from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

# from ray.rllib.algorithms.ppo import PPOConfig
from stable_baselines3.common.callbacks import BaseCallback
from chop.passes.analysis.total_bits_estimator import total_bits_module_analysis_pass


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
        cost = total_bits_module_analysis_pass(model, {})
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


class MixedPrecisionEnv(gym.Env):
    def __init__(self, config):
        # Make the space (for actions and observations) configurable.
        # Since actions should repeat observations, their spaces must be the
        # same.
        search_space = config.get("search_space", None)
        runner = config.get("runner", None)
        self.search_space, self.runner = search_space, runner
        if search_space is None:
            raise ValueError(f"search_space cannot be None")

        # observation space definition
        self.observation_space = Dict({"loss": Box(0.0, 10e4, shape=(1,))})
        for k, v in search_space.search_space_flattened.items():
            k1, k2 = k.split(".")
            new_k = "_and_".join([k1, k2])
            self.observation_space[new_k] = Discrete(v)
        my_list = list(search_space.per_layer_search_space.values())

        # action space definition
        self.action_space = MultiDiscrete((search_space.num_layers, *my_list))

        self.cur_obs = None
        self.episode_len = 0

    def get_performance(self, sample):
        return self.runner.get_performance(
            sample, self.search_space, self.runner.runner
        )

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
        loss = self.cur_obs.pop("loss")
        cur_obs = {k: v for k, v in self.cur_obs.items()}
        sample, sample_idx = self.search_space.build_sample(cur_obs, "_and_")

        # We use the first discrete value in action to pick layer
        # The rest of the action discrete values
        # are used to pick quantisation choices in that layer
        i = 0
        for k, v in sample.items():
            if i == action[0]:
                j = 1
                for k2, v in v["config"].items():
                    if k2 == "name":
                        sample[k]["config"][k2] = v
                    else:
                        choices = self.search_space.choices[k][k2]
                        sample[k]["config"][k2] = choices[action[j]]
                        sample_idx[k][k2] = action[j]
                        j += 1

        # get performance
        metric = self.get_performance(sample)
        metric = metric.detach().numpy()

        # Set `truncated` flag after 10 steps.
        self.episode_len += 1
        terminated = False
        truncated = self.episode_len >= 10

        # TODO: this reward needs redesign, apparently
        reward = -metric
        # Set a new observation (random sample).
        rebuild_sample = {}
        for k, v in sample_idx.items():
            for k2, v2 in v.items():
                rebuild_sample[f"{k}_and_{k2}"] = v2

        rebuild_sample["loss"] = [metric]
        self.cur_obs = rebuild_sample
        return self.cur_obs, reward, terminated, truncated, {"loss": metric}


env_map = {
    "mixed_precision": MixedPrecisionEnv,
    "llm_mixed_precision": LLMMixedPrecisionEnv,
}
