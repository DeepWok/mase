from ..base import SearchStrategyBase
from .env import env_map
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

algorithm_map = {
    "ppo": PPO,
    "a2c": A2C,
}

class StrategyRL(SearchStrategyBase):
    iterative = True

    def _post_init_setup(self):
        self.algorithm_name = self.config["algorithm"]
        self.algorithm = algorithm_map[self.algorithm_name]
        self.total_timesteps = self.config["total_timesteps"]
        self.save_name = self.config["save_name"]
        self.env = env_map[self.config["env"]]
        self.device = self.config["device"]
        self.episode_max_len = 10 # TODO: Try and change this
    
    def search(self, search_space):
        env = self.env(config=self.config, search_space=search_space, sw_runner=self.sw_runner, hw_runner=self.hw_runner, data_module=self.data_module, episode_max_len=self.episode_max_len)

        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/")
        eval_callback = EvalCallback(
            env,
            best_model_save_path="./logs/best_model",
            log_path="./logs/results",
            eval_freq=100,
        )
        callback = CallbackList([checkpoint_callback, eval_callback])

        # possible extension is to allow custom policy network
        # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

        model = self.algorithm(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=self.device,
            tensorboard_log="./logs/",
            n_steps=64, # TODO: Try and change this
        )

        vec_env = model.get_env()
        model.learn(
            total_timesteps=int(self.total_timesteps),
            progress_bar=True,
            callback=callback,
        )

        # TODO
        # drop this to mase_output
        model.save(self.save_name)

        # inference run, but not needed?
        obs = vec_env.reset()
        for _ in range(self.episode_max_len):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

        print(obs) # TODO: Pretty print results
        
        return obs["cost"], obs, model