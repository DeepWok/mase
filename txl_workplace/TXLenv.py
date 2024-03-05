import gymnasium as gym

from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete


class MixedPrecisionEnv(gym.Env):
    def __init__(self, config):
        # Make the space (for actions and observations) configurable.
        # Since actions should repeat observations, their spaces must be the
        # same.
        self.search_space = config.get("search_space", None)
        self.run_trial = config.get("run_trial", None)

        # TODO get layer information from self.search_space.model to build observation space
        # TODO get choices from self.search_space.choices_flattened to build action space
        # observation space definition
        self.observation_space = Dict({"loss": Box(0.0, 10e4, shape=(1,))})
        for k, v in self.search_space.search_space_flattened.items():
            k1, k2 = k.split(".")
            new_k = "_and_".join([k1, k2])
            self.observation_space[new_k] = Discrete(v)
        my_list = list(self.search_space.per_layer_search_space.values())

        # action space definition
        self.action_space = MultiDiscrete((self.search_space.num_layers, *my_list))

        self.cur_obs = None
        self.episode_len = 0

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
        metric = self.run_trial(sample)

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