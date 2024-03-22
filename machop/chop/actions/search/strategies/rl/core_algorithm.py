from ..base import SearchStrategyBase
from .env import env_map
from pprint import pprint
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

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
        self.device = self.config["device"]
        self.episode_max_len = 10  # This might be adjusted based on trials
        
    def make_env(self, env_id, rank, seed=0):
        def _init():
            env = env_map[env_id](config=self.config, search_space=self.search_space, sw_runner=self.sw_runner, hw_runner=self.hw_runner, data_module=self.data_module, episode_max_len=self.episode_max_len)
            env.seed(seed + rank)
            return env
        return _init

    def search(self, search_space):
        self.search_space = search_space  # Storing search_space for use in make_env
        
        if 'load_name' in self.config:
            # Assuming that loading the model correctly handles a vectorized environment
            model = self.algorithm.load(self.config['load_name'])
            print(f"Model loaded from {self.config['load_name']}. Skipping training.")
        else:
            num_envs = 4  # Define the number of environments to run in parallel
            env_id = self.config["env"]
            #envs = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_envs)])
            envs = envs = SubprocVecEnv([self.make_env(env_id, i) for i in range(num_envs)], start_method='spawn')
            
            checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/")
            eval_callback = EvalCallback(
                envs,
                best_model_save_path="./logs/best_model",
                log_path="./logs/results",
                eval_freq=100,
            )
            callback = CallbackList([checkpoint_callback, eval_callback])
            
            model = self.algorithm(
                "MultiInputPolicy",
                envs,
                verbose=1,
                device=self.device,
                tensorboard_log="./logs/",
                n_steps=256,  # This might be adjusted based on trials
            )
            
            model.learn(
                total_timesteps=int(self.total_timesteps),
                progress_bar=True,
                callback=callback,
            )
            
            model.save(self.save_name)
            print(f"Model trained and saved as {self.save_name}.")

        # Evaluating the model using the same vectorized environment
        vec_env = model.get_env()
        obs = vec_env.reset()
        for _ in range(self.episode_max_len):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
        
        # Assuming the observation has a "cost" field for the purpose of this example
        costs = [ob["cost"] for ob in obs]  # You might need to adjust this based on your environment's observations
        pprint(obs)
        
        # Return average cost and the last observations and model as an example
        avg_cost = sum(costs) / len(costs)
        return avg_cost, obs, model
