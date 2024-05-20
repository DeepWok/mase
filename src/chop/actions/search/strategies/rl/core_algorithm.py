import wandb
from ..base import SearchStrategyBase
from .env import env_map, registered_env_map
from pprint import pprint
import numpy as np
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
import json
import warnings

warnings.filterwarnings("ignore")

algorithm_map = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG}


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals.get("infos")[-1]
        wandb.log(info)
        return True


class SearchStrategyRL(SearchStrategyBase):
    iterative = True

    def _post_init_setup(self):
        # Default configuration values
        defaults = {
            "algorithm": "ppo",
            "env": "mixed_precision_paper",
            "device": "cuda",
            "total_timesteps": 100000,
            "n_steps": 32,
            "n_envs": 4,
            "eval_freq": 200,
            "save_freq": 200,
            "episode_max_len": 1000,
            "learning_rate": 2.5e-4,
            "save_name": "tmp_rl",
            "wandb_callback": False,
            "wandb_entity": "",
        }

        # Get configuration values from the config dictionary or use defaults
        self.algorithm_name = self.config.get("algorithm", defaults["algorithm"])
        self.env_name = self.config.get("env", defaults["env"])
        self.device = self.config.get("device", defaults["device"])
        self.total_timesteps = self.config.get(
            "total_timesteps", defaults["total_timesteps"]
        )
        self.n_steps = self.config.get("n_steps", defaults["n_steps"])
        self.n_envs = (
            self.config.get("n_envs", defaults["n_envs"])
            if self.algorithm_name != "ddpg"
            else 1
        )
        self.eval_freq = self.config.get("eval_freq", defaults["eval_freq"])
        self.save_freq = self.config.get("save_freq", defaults["save_freq"])
        self.episode_max_len = self.config.get(
            "episode_max_len", defaults["episode_max_len"]
        )
        self.learning_rate = self.config.get("learning_rate", defaults["learning_rate"])
        self.save_name = self.config.get("save_name", defaults["save_name"])
        self.wandb_callback = self.config.get(
            "wandb_callback", defaults["wandb_callback"]
        )
        self.wandb_entity = self.config.get("wandb_entity", defaults["wandb_entity"])

        # Get environment and algorithm based on the names
        self.env = env_map[self.env_name]
        self.registered_env_name = registered_env_map[self.env_name]
        self.algorithm = algorithm_map[self.algorithm_name]

    def _create_env(self, search_space):
        # Create the vectorized environment
        env = make_vec_env(
            self.registered_env_name,
            n_envs=self.n_envs,
            seed=0,
            env_kwargs={
                "config": self.config,
                "search_space": search_space,
                "sw_runner": self.sw_runner,
                "hw_runner": self.hw_runner,
                "data_module": self.data_module,
                "episode_max_len": self.episode_max_len,
            },
        )
        return env

    def _initialize_callbacks(self, env):
        callbacks = []

        # Add WandbCallback if enabled
        if self.wandb_callback:
            wandb.init(project="Mase-RL", entity=self.wandb_entity)
            wandb_callback = WandbCallback()
            callbacks.append(wandb_callback)

        # Add CheckpointCallback and EvalCallback
        save_path = self.save_dir / "rl_logs"
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq, save_path=save_path
        )
        eval_callback = EvalCallback(
            env,
            best_model_save_path=save_path / "rl_best_model",
            log_path=save_path / "rl_results",
            eval_freq=self.eval_freq,
        )
        callbacks.extend([checkpoint_callback, eval_callback])
        return CallbackList(callbacks)

    def search(self, search_space):
        env = self._create_env(search_space)
        callback = self._initialize_callbacks(env)

        if "rl_model_load_name" in self.config:
            # Load pre-trained model if specified
            model = self.algorithm.load(
                self.config["rl_model_load_name"],
                env=self.env(
                    config=self.config,
                    search_space=search_space,
                    sw_runner=self.sw_runner,
                    hw_runner=self.hw_runner,
                    data_module=self.data_module,
                    episode_max_len=self.episode_max_len,
                ),
            )
            print(
                f"Model loaded from {self.config['rl_model_load_name']}. Skipping training."
            )

        else:
            algorithm_kwargs = {
                "verbose": 1,
                "device": self.device,
                "tensorboard_log": self.save_dir / "tb_logs",
                "learning_rate": self.learning_rate,
            }

            if self.algorithm_name != "ddpg":
                algorithm_kwargs["n_steps"] = self.n_steps

            policy = (
                "MlpPolicy"
                if self.config["env"] == "mixed_precision_paper"
                else "MultiInputPolicy"
            )

            # Create and train the model
            model = self.algorithm(policy, env, **algorithm_kwargs)

            model.learn(
                total_timesteps=int(self.total_timesteps),
                progress_bar=True,
                callback=callback,
            )

            # Save the model after training
            model.save(self.save_name)
            print(f"Model trained and saved as {self.save_name}.")

        self.n_envs = 1
        # Actual Prediction
        vec_env = make_vec_env(
            self.registered_env_name,
            n_envs=self.n_envs,
            seed=0,
            env_kwargs={
                "config": self.config,
                "search_space": search_space,
                "sw_runner": self.sw_runner,
                "hw_runner": self.hw_runner,
                "data_module": self.data_module,
                "episode_max_len": self.episode_max_len,
            },
        )

        n_envs = len(vec_env.envs)
        dones = np.full(n_envs, False, dtype=bool)
        obs = vec_env.reset()
        while not dones.all():
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)

        print("The best configurations found is: ")
        pprint(vec_env.get_attr("best_sample"))

        print("\nThe corresponding metrics are: ")
        pprint(vec_env.get_attr("best_performance"))

        data = {
            "best_configurations": vec_env.get_attr("best_sample"),
            "metrics": vec_env.get_attr("best_performance"),
        }

        file_path = self.save_dir / "rl_output.json"

        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

        print(f"Data has been saved to {file_path}")
