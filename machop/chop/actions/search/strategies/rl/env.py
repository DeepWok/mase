import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.utils import get_mase_op, get_node_actual_target
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
)

class MixedPrecisionEnv(gym.Env):
    def __init__(self, config):
        self.search_space = config.get("search_space", None)
        self.run_trial = config.get("run_trial", None)

        # observation space definition
        # get layer information from self.search_space.model
        graph = MaseGraph(self.search_space.model)
        graph, _ = init_metadata_analysis_pass(graph)
        graph, _ = add_common_metadata_analysis_pass(graph, {"dummy_in": self.search_space.dummy_input})
        layer_info = {}
        idx = 0
        for node in graph.fx_graph.nodes:
            if get_mase_op(node) == 'linear':
                target = get_node_actual_target(node)
                layer_info[node.name] = [idx, target.in_features, target.out_features, 1, 0]
                idx += 1
            elif get_mase_op(node) == 'conv2d':
                target = get_node_actual_target(node)
                layer_info[node.name] = [idx, target.in_channels, target.out_channels, target.kernel_size[0], target.stride[0]]
                idx += 1
            # elif get_mase_op(node) == 'relu':
            #     layer_info[node.name] = [idx, 0, 0, 1, 0]
            #     idx += 1
        # get choices from self.search_space.choices_flattened to build observation enumeration list
        self.obs_list = []
        self.act_list = []
        self.sample_namespace = []
        self.sample = {}
        for name, choices in self.search_space.choices_flattened.items():
            if len(choices) == 1:
                self.sample[name] = 0
                continue
            self.sample_namespace.append(name)
            _name = name.split('/')
            obs = layer_info[_name[0]].copy()
            if _name[2] == 'data_in_width':
                obs.append(0)
            elif _name[2] == 'weight_width':
                obs.append(1)
            elif _name[2] == 'bias_width':
                obs.append(2)
            self.obs_list.append(obs)
            self.act_list.append(sorted(choices))

        self.state = 0
        self.obs_list = np.array(self.obs_list)
        # get observation space definition from observation enumeration list
        low = np.min(self.obs_list, axis=0)
        high = np.max(self.obs_list, axis=0)
        self.observation_space = Box(low=np.append(low, min([min(sub) for sub in self.act_list])), high=np.append(high, max([max(sub) for sub in self.act_list])))
        self.action_space = Box(low=0, high=1.)

    def reset(self, *, seed=None, options=None):
        """
            Resets the episode and returns the initial observation of the new one.
            Always start from the first element in observation list.
        """
        self.state = 0
        obs = np.append(self.obs_list[self.state, :], min(self.act_list[self.state])).astype(np.float32)
        return obs, {}

    def step(self, action):
        """Takes a single step in the episode given `action`
            The episode would end in fixed timestep (same with the length of observation list)
        Returns:
            observation (ObsType): A list format as [order of layer, input channels, output channels, kernel size, stride size, data/weight/bias, previous action].
            reward (SupportsFloat): Sum of metrics calculated by provided function in SearchStrategy.
            terminated (bool): .
            truncated (bool): Always False. No need for truncation, as the episode is fixed.
            info (dict): Empty.
        """
        choices = self.act_list[self.state]
        action = int(action*len(choices) - 1e-2)
        self.sample[self.sample_namespace[self.state]] = action
        reward = 0
        terminated = False
        self.state += 1
        if self.state == len(self.obs_list):
            self.state = 0
            terminated = True
            reward = self.run_trial(self.sample)
        obs = self.obs_list[self.state].copy()
        obs = np.append(obs, choices[action]).astype(np.float32)
        # TODO delete later
        if reward != 0:
            with open('./txl_workplace/log.txt', 'a') as f:
                f.write(f'{reward}\n')
        return obs, reward, terminated, False, {}