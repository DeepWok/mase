import gymnasium as gym
import copy
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from stable_baselines3.common.callbacks import BaseCallback

class MixedPrecisionEnv(gym.Env):
    def __init__(self, config):
        pass