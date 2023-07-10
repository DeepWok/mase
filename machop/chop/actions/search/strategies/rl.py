import gymnasium as gym

from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from .base import StrategyBase

# from ray.rllib.algorithms.ppo import PPOConfig
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback


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
        self.observation_space = Dict({"loss": Box(0.0, 10e4, shape=(1,))})

        for k, v in search_space.search_space_flattened.items():
            k1, k2 = k.split(".")
            new_k = "_and_".join([k1, k2])
            self.observation_space[new_k] = Discrete(v)
        print(self.observation_space)
        my_list = list(search_space.per_layer_search_space.values())
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
}

algorithm_map = {"ppo": PPO, "a2c": A2C}


class StrategyRL(StrategyBase):
    iterative = True

    def read_config(self):
        self.algorithm_name = self.config["algorithm"]

        self.device = self.config["device"]
        self.total_timesteps = self.config["total_timesteps"]
        self.save_name = self.config["save_name"]

        self.env_name = self.config["env"]
        self.env = env_map[self.env_name]
        self.algorithm = algorithm_map[self.algorithm_name]

    def search(self, search_space, runner):
        env = MixedPrecisionEnv(config={"search_space": search_space, "runner": runner})

        # TODO
        # possible extension is to allow custom policy network
        # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

        model = self.algorithm("MultiInputPolicy", env, verbose=1, device=self.device)
        vec_env = model.get_env()
        model.learn(total_timesteps=int(self.total_timesteps), progress_bar=True)
        # TODO
        # improvements needed
        # drop this to mase_output
        model.save(self.save_name)

        # inference run, but not needed?
        obs = vec_env.reset()
        for _ in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
        return obs["loss"], obs, model
