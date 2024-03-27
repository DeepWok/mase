# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
import copy
from torch import nn
import torch
from torch.nn import ReLU
from ..base import SearchSpaceBase
from .....passes.graph.transforms.quantize import (
    QUANTIZEABLE_OP,
    quantize_transform_pass,
)
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target, get_parent_name
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict

from nas_201_api import NASBench201API as API
from .xautodl.models import get_cell_based_tiny_net

from .model_spec import ModelSpec

# from nas_graph import load_bench_arch

'''
Group 2 NAS-Proxy
This file defines the search space for mixed-precision post-training-quantization quantization search on mase graph.

Imports:
- Various utility functions and classes are imported from different modules.

Constants:
- DEFAULT_ZERO_COST_ARCHITECTURE_CONFIG: This is the default architecture configuration for the zero-cost NAS.

Classes:
- ZeroCostProxy: This class defines the search space for post-training quantization on mase graph.

Functions:
- instantiate_linear: This function creates an instance of a linear layer with the given parameters.
- instantiate_relu: This function creates an instance of a ReLU activation function with the given parameters.
- instantiate_batchnorm: This function creates an instance of a batch normalization layer with the given parameters.
- instantiate_conv2d: This function creates an instance of a 2D convolutional layer with the given parameters.
- generate_configs: This function generates the configuration for the NAS based on the given dictionary.
'''

### default architecture is the architecuture returned 
### by api.get_net_config(0, 'cifar10') in nasbench201
DEFAULT_ZERO_COST_ARCHITECTURE_CONFIG = {
    "config": {'dataset': 'cifar10',
    'name': 'infer.tiny', 
    'C': 16, 
    'N': 5, 
    'op_0_0': 0, 
    'op_1_0': 4, 
    'op_2_0': 2, 'op_2_1': 1, 
    'op_3_0': 2, 'op_3_1': 1, 'op_3_2': 1, 
    'number_classes': 10}
}

print("loading api")
api = API('./third_party/NAS-Bench-201-v1_1-096897.pth', verbose=False)
print("api loaded")

class ZeroCostProxy(SearchSpaceBase):
    """
    Group 2 NAS-Proxy
    Post-Training quantization search space for mase graph.
    This class uses the NAS-Bench-201 API to query the architecture performance and uses the returned architecture for further operations.
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_ZERO_COST_ARCHITECTURE_CONFIG

    def rebuild_model(self, sampled_config, is_eval_mode: bool = False):
        '''
        Group 2 NAS-Proxy
        This method rebuilds the model based on the sampled configuration. It also sets the model to evaluation or training mode based on the is_eval_mode parameter.
        It queries the NAS-Bench-201 API for the architecture performance and uses the returned architecture to rebuild the model.
        '''
        print("\n=========sampled_config===============")
        print(sampled_config)
        
        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()
       
        if "nas_zero_cost" in sampled_config:
            nas_config = generate_configs(sampled_config["nas_zero_cost"])
            nasbench_dataset = sampled_config["nas_zero_cost"]["dataset"]
        else:
            nas_config = generate_configs(sampled_config["default"])
            nasbench_dataset = sampled_config["default"]["dataset"]

        print(f"Used dataset: {nasbench_dataset}")
        
        arch = nas_config["arch_str"]
        index = api.query_index_by_arch(arch)
        print("index")
        print(index)
        results = api.query_by_index(index, nasbench_dataset)
        print("results")
        print(results)
        data = api.get_more_info(index, nasbench_dataset)
        
        model_arch = get_cell_based_tiny_net(nas_config)
        model_arch = model_arch.to(self.accelerator)

        return model_arch, data

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for zero-cost
        """
        
        choices = {}
        choices["nas_zero_cost"] = self.config["nas_zero_cost"]["config"]

        for key, value in DEFAULT_ZERO_COST_ARCHITECTURE_CONFIG["config"].items():
            if key in choices["nas_zero_cost"]:
                continue
            else:
                choices["nas_zero_cost"][key] = value
        
        # flatten the choices and choice_lengths
        # self.choices_flattened = {}
        flatten_dict(choices, flattened=self.choices_flattened)
        
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }

        print(self.choice_lengths_flattened)
        

    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        flattened_config = {}
        # import pdb; pdb.set_trace() 
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        return config
    
    # def build_search_space(self, config_all):
    #     # choises = {}
        
    #     self.choices_flattened = generate_configs(config_all)
        
    #     self.choice_lengths_flattened = {
    #         k: len(v) for k, v in self.choices_flattened.items()
    #     }

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def instantiate_relu(inplace):
    return ReLU(inplace)

def instantiate_batchnorm(num_features, eps, momentum, affine, track_running_stats):
    return nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

def instantiate_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
        bias = bias,
        padding_mode = 'same',
        device = None,
        dtype = None,
    )

import itertools

def generate_configs(config_dict):
    '''
    Group 2 NAS-Proxy
    This function generates the configuration for the NAS based on the given dictionary.
    This is used to generate the configuration that is used to query the NAS-Bench-201 API for the architecture performance.
    '''
    name = config_dict['name']
    C = config_dict['C']
    N = config_dict['N']
    num_classes = config_dict['number_classes']
    op_map = {0:'skip_connect', 1:'none', 2:'nor_conv_3x3', 3:'nor_conv_1x1', 4:'avg_pool_3x3'}

    ### generate combination
    arch_str = ""
    for target_neuro in range(1, 4):
        arch_str += "|"
        for exert_neuro in range(0, target_neuro):
            op = f"op_{target_neuro}_{exert_neuro}"
            op_str = op_map[config_dict[op]]
            op_str += f"~{exert_neuro}"
            arch_str += (op_str + "|")
        if target_neuro < 3:
            arch_str += "+"
    
    config = {
        'name': name,
        'C': C,
        'N': N,
        'arch_str': arch_str,
        'num_classes': num_classes
    }

    return config