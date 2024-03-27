import re
from xautodl.models import get_cell_based_tiny_net
from nas_201_api import NASBench201API as API
from nas_201_api.api_utils import ArchResults

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
OP_NAMES_NB201 = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

def get_ops_from_arch(arch):
    nodes = arch.split('+')
    def get_op(x):
        return x.split('~')[0]
    
    def get_parent(x):
        return x.split('~')[1]

    node_ops = [list(map(get_op, n.strip()[1:-1].split('|'))) for n in nodes]
    
    node_parents = [list(map(get_parent, n.strip()[1:-1].split('|'))) for n in nodes]

    enc = []
    for u, v in EDGE_LIST:
        enc.append(OP_NAMES_NB201.index(node_ops[v-2][u-1]))

    return tuple(enc), node_ops, node_parents

def get_config_of_network(config_nodes_index):
    architecture_config = {}

    for node_index, ops in enumerate(config_nodes_index):
        inputs = []
        for input_index, connection in enumerate(OP_NAMES_NB201[node_index]):
            if connection:
                inputs.append(f'node{connection-1}')
        
        architecture_config[f'node{node_index}'] = {'inputs': inputs, 'operations': ops}

    return architecture_config

def get_ops_and_parent_node(arch, arch_index):
    print("loading api")
    api = API('/home/xz2723/mase_xinyi/machop/third_party/NAS-Bench-201-v1_1-096897.pth', verbose=False)
    print("api loaded")
    config = api.get_net_config(arch_index, 'cifar10')
    config_nodes, node_ops, node_parents = get_ops_from_arch(config['arch_str'])
    
    return config_nodes, node_ops, node_parents

print("loading api")
api = API('/home/xz2723/mase_xinyi/machop/third_party/NAS-Bench-201-v1_1-096897.pth', verbose=False)
print("api loaded")
config = api.get_net_config(0, 'cifar10')

print(config)
print(config.keys())
print(config['name'])
print(config['arch_str'])

# import pdb; pdb.set_trace()

config_nodes, node_ops, node_parents = get_ops_from_arch(config['arch_str'])
print(config_nodes)
print(node_ops)
print(node_parents)

# config_f = get_config_of_network(config_nodes)
# print(config_f)
param = api.str2matrix(config["arch_str"])
print(param)

param_list = api.str2lists(config["arch_str"])
print(param_list)

network = get_cell_based_tiny_net(config)
print("network:", network)

print(network['cells'])