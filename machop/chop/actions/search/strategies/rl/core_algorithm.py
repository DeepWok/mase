from ..base import StrategyBase
from .env import env_map
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)


algorithm_map = {
    # TODO: maybe network architecture needs complication.
    "ppo": PPO,
    "a2c": A2C,
}


class StrategyRL(StrategyBase):
    iterative = True

    def read_setup(self):
        setup = self.config["setup"]
        self.model_parallel = setup["model_parallel"]
        self.runner_style = setup["runner_style"]
        self.runner = self.get_runner(self.runner_style)

        self.algorithm_name = setup["algorithm"]
        # self.device = setup["device"]
        self.total_timesteps = setup["total_timesteps"]
        self.save_name = setup["save_name"]

        self.env_name = setup["env"]
        self.env = env_map[self.env_name]
        self.algorithm = algorithm_map[self.algorithm_name]

    def search(self, search_space):
        env = self.env(config={"search_space": search_space, "runner": self.runner})

        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/")
        eval_callback = EvalCallback(
            env,
            best_model_save_path="./logs/best_model",
            log_path="./logs/results",
            eval_freq=500,
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
        )

        vec_env = model.get_env()
        model.learn(
            total_timesteps=int(self.total_timesteps),
            progress_bar=True,
            callback=callback,
        )

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
