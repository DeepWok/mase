from ..base import SearchStrategyBase
from .env import env_map, registered_env_map
from pprint import pprint
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
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
        self.n_steps = self.config["n_steps"]
        self.n_envs = self.config["n_envs"]
        self.eval_freq = self.config["eval_freq"]
        self.save_freq = self.config["save_freq"]
        self.episode_max_len = self.config["episode_max_len"]
        self.learning_rate = self.config["learning_rate"]
        self.registered_env_name = registered_env_map[self.config["env"]]

    def search(self, search_space):
        # Check if load_name is specified in config and load the model if so
        if 'load_name' in self.config:
            model = self.algorithm.load(self.config['load_name'], env=self.env(config=self.config, search_space=search_space, sw_runner=self.sw_runner, hw_runner=self.hw_runner, data_module=self.data_module, episode_max_len=self.episode_max_len))
            print(f"Model loaded from {self.config['load_name']}. Skipping training.")
        else:
            #env =self.env(config=self.config, search_space=search_space, sw_runner=self.sw_runner, hw_runner=self.hw_runner, data_module=self.data_module, episode_max_len=self.episode_max_len)
            env = make_vec_env(
                self.registered_env_name, n_envs=self.n_envs, seed=0, 
                env_kwargs={"config": self.config, "search_space": search_space , 
                "sw_runner": self.sw_runner, "hw_runner": self.hw_runner, 
                "data_module": self.data_module, "episode_max_len":self.episode_max_len}
                )

            checkpoint_callback = CheckpointCallback(save_freq=self.save_freq, save_path="./logs/")
            eval_callback = EvalCallback(
                env,
                best_model_save_path="./logs/best_model",
                log_path="./logs/results",
                eval_freq=self.eval_freq,
            )
            callback = CallbackList([checkpoint_callback, eval_callback])

            model = self.algorithm(
                "MultiInputPolicy",
                env,
                verbose=1,
                device=self.device,
                tensorboard_log="./logs/",
                n_steps=self.n_steps,
                learning_rate=self.learning_rate,
            )

            vec_env = model.get_env()
            model.learn(
                total_timesteps=int(self.total_timesteps),
                progress_bar=True,
                callback=callback,
            )

            # Save the model after training
            model.save(self.save_name)
            print(f"Model trained and saved as {self.save_name}.")

        # Post-training or post-loading actions
        vec_env = model.get_env()
        obs = vec_env.reset()
        for _ in range(self.episode_max_len):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

        pprint(obs)

        return obs["cost"], obs, model
