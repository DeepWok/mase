import time
import torch
from ..base import SearchStrategyBase
from .env import env_map
from stable_baselines3 import A2C, PPO, DDPG, TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
import logging

logger = logging.getLogger(__name__)

algorithm_map = {"ppo": PPO, "a2c": A2C, "td3": TD3}


class StrategyRL(SearchStrategyBase):
    iterative = True

    def _post_init_setup(self):
        setup = self.config["setup"]
        self.model_parallel = setup["model_parallel"]
        self.runner_style = setup["runner_style"]

        self.algorithm_name = setup["algorithm"]
        self.device = setup["device"]
        self.total_timesteps = setup["total_timesteps"]
        # self.save_name = "./mase_output/"+setup["save_name"]+'/'+setup["save_name"]
        self.save_name = (
                "./mase_output/" + setup["save_name"] + "/" + setup["save_name"]
        )
        self.env_name = setup["env"]
        self.env = env_map[self.env_name]
        self.algorithm = algorithm_map[self.algorithm_name]
        self.sum_scaled_metrics = setup["sum_scaled_metrics"]

        self.metrics = self.config["metrics"]
        self.load_path = setup["load_path"]

        self.mode = setup["mode"]

    def search(self, search_space):
        # env = self.env(config={"search_space": search_space, "runner": self.runner})
        # env = self.env(config={"search_space": search_space,
        #                        "hw_runner": self.hw_runner,
        #                        "sw_runner": self.sw_runner,
        #                        "sum_scaled_metrics": self.sum_scaled_metrics,
        #                        "data_module": self.data_module,
        #                        "metrics": self.metrics})

        env = self.env(
            config={
                "search_space": search_space,
                "hw_runner": self.hw_runner,
                "sw_runner": self.sw_runner,
                "sum_scaled_metrics": self.sum_scaled_metrics,
                "data_module": self.data_module,
                "metrics": self.metrics,
            },
            verbose=0
        )
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/")
        eval_callback = EvalCallback(
            env,
            best_model_save_path="./logs/best_model",
            log_path="./logs/results",
            eval_freq=1000,
        )
        callback = CallbackList([checkpoint_callback, eval_callback])
        method = 0
        # possible extension is to allow custom policy network
        # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
        if self.mode == "train":
            logger.info("training from sketch")
            model = self.algorithm(
                "MultiInputPolicy",
                env,
                verbose=1,
                device=self.device,
                tensorboard_log="./logs/",
            )

            vec_env = model.get_env()
            start = time.time()
            print(int(self.total_timesteps))
            model.learn(
                total_timesteps=int(self.total_timesteps),
                callback=callback,
            )
            print(time.time() - start)
            logger.info("Finished training")
            model.save(self.save_name)
            obs = vec_env.reset()
            for _ in range(1):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
            return obs["reward"], obs, model
        elif self.mode == "continue-training":
            logger.info("Continue training")
            # Continue Training
            model = self.algorithm.load(self.load_path, device=self.device, env=env)

            vec_env = model.get_env()

            model.learn(
                total_timesteps=int(self.total_timesteps),
                callback=callback,
            )
            obs = vec_env.reset()
            for _ in range(1):
                action, _state = model.predict(obs, deterministic=True)
                logger.info(action)
                obs, reward, done, info = vec_env.step(action)
                print(obs["reward"])
            return obs["reward"], obs, model
        elif self.mode == "load":
            logger.info("Loading RL model")
            model = self.algorithm.load(self.load_path, device=self.device, env=env)
            vec_env = model.get_env()
            obs = vec_env.reset()
            for _ in range(20):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
            logger.info(
                f"Best model: Accuracy={env.result[0]}, Average Bit Width={env.result[1]}"
            )
            return obs["reward"], obs, model
        else:
            logger.warning(f"{self.mode} not implemented")
            return None
