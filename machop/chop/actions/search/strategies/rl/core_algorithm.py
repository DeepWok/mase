import wandb
from ..base import SearchStrategyBase
from .env import env_map, registered_env_map
from pprint import pprint
from stable_baselines3 import A2C, PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback

algorithm_map = {
    "ppo": PPO,
    "a2c": A2C,
    "ddpg": DDPG,
    "sac": SAC,
}

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals.get('infos')[-1]
        wandb.log(info)
        return True


class StrategyRL(SearchStrategyBase):
    iterative = True
    
    def _post_init_setup(self):
        defaults = {
            "algorithm": 'ppo',
            "env": 'mixed_precision_paper',
            "device": 'cuda',
            "total_timesteps": 100000,
            "n_steps": 32,
            "n_envs": 4,
            "eval_freq": 200,
            "save_freq": 200,
            "episode_max_len": 10,
            "learning_rate": 2.5e-4,
            "save_name": 'tmp_rl',
        }
        
        self.algorithm_name = self.config.get("algorithm", defaults["algorithm"])
        self.env_name = self.config.get("env", defaults["env"])
        self.device = self.config.get("device", defaults["device"])
        self.total_timesteps = self.config.get("total_timesteps", defaults["total_timesteps"])
        self.n_steps = self.config.get("n_steps", defaults["n_steps"])
        self.n_envs = self.config.get("n_envs", defaults["n_envs"])
        self.eval_freq = self.config.get("eval_freq", defaults["eval_freq"])
        self.save_freq = self.config.get("save_freq", defaults["save_freq"])
        self.episode_max_len = self.config.get("episode_max_len", defaults["episode_max_len"])
        self.learning_rate = self.config.get("learning_rate", defaults["learning_rate"])
        self.save_name = self.config.get("save_name", defaults["save_name"])

        self.env = env_map[self.env_name]
        self.registered_env_name = registered_env_map[self.env_name]
        self.algorithm = algorithm_map[self.algorithm_name]

    def _create_env(self, search_space):
        env = make_vec_env(
            self.registered_env_name, n_envs=self.n_envs, seed=0, 
            env_kwargs={"config": self.config, "search_space": search_space , 
            "sw_runner": self.sw_runner, "hw_runner": self.hw_runner, 
            "data_module": self.data_module, "episode_max_len":self.episode_max_len}
            )
        return env

    def _initialize_callbacks(self, env):
        wandb.init(project="Mase-RL", entity="m-pl-braganca")
        wandb_callback = WandbCallback()
        checkpoint_callback = CheckpointCallback(save_freq=self.save_freq, save_path="./logs/")
        eval_callback = EvalCallback(
            env,
            best_model_save_path="./logs/best_model",
            log_path="./logs/results",
            eval_freq=self.eval_freq,
        )
        callbacks = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
        return callbacks

    def search(self, search_space):
        if 'load_name' in self.config:
            model = self.algorithm.load(self.config['load_name'], env=self.env(config=self.config, search_space=search_space, sw_runner=self.sw_runner, hw_runner=self.hw_runner, data_module=self.data_module, episode_max_len=self.episode_max_len))
            print(f"Model loaded from {self.config['load_name']}. Skipping training.")

        else:
            env = self._create_env(search_space)
            callback = self._initialize_callbacks(env)

            algorithm_kwargs = {
                "verbose": 1,
                "device": self.device,
                "tensorboard_log": "./logs/",
                "learning_rate": self.learning_rate,
            }

            if self.algorithm_name != "ddpg":
                algorithm_kwargs["n_steps"] = self.n_steps

            policy = "MlpPolicy" if self.config["env"] == "mixed_precision_paper" else "MultiInputPolicy"

            model = self.algorithm(
                policy,
                env,
                **algorithm_kwargs
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