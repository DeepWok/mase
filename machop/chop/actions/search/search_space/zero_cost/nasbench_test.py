from nas_201_api import NASBench201API as API
from nas_201_api.api_utils import ArchResults
from xautodl.models import get_cell_based_tiny_net
# Create an API without the verbose log
print("loading api")
api = API('/home/xz2723/mase_xinyi/machop/third_party/NAS-Bench-201-v1_1-096897.pth', verbose=False)
print("api loaded")

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
OP_NAMES_NB201 = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
OPS_TO_NB201 = {
    "AvgPool1x1": "avg_pool_3x3",
    "ReLUConvBN1x1": "nor_conv_1x1",
    "ReLUConvBN3x3": "nor_conv_3x3",
    "Identity": "skip_connect",
    "Zero": "none",
}

def convert_str_to_op_indices(str_encoding):
    """
    Converts NB201 string representation to op_indices
    """
    nodes = str_encoding.split('+')
    def get_op(x):
        return x.split('~')[0]

    node_ops = [list(map(get_op, n.strip()[1:-1].split('|'))) for n in nodes]

    enc = []
    for u, v in EDGE_LIST:
        enc.append(OP_NAMES_NB201.index(node_ops[v-2][u-1]))

    return tuple(enc)

# x = api.query_meta_info('cifar10', 'x-valid')
config = api.get_net_config(12, 'cifar10')
print(config)

# network = get_cell_based_tiny_net(config)
param = api.str2matrix(config["arch_str"])
print(param)

param_list = api.str2lists(config["arch_str"])
print(param_list)

# info = api.get_more_info(123, 'cifar10', None, hp="200", is_random=True)
# print(info)

# print("arch:")
# results = api.query_by_index(1, 'cifar10')
# print(results)

# config = api.get_net_config(123, 'cifar10') # obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
network = get_cell_based_tiny_net(config) # create the network from configurration
print("network:", network)
# print("config:", config)
# print("config type:", type(config))
# print("type", type(network))

# weights = api.get_net_param(3, 'cifar10', None)

# config = api.get_net_config(3, 'cifar10')
# network2 = get_cell_based_tiny_net(config)
# print("network2:", network2)
# print("type", type(network2))

# print("try best")
# arch_index, accuracy = api.find_best('cifar10', 'x-valid', None, None, '12')
# api.show(arch_index)

# print(network2)
# print(type(network2.cells))
# # print(network2.arch_str)
# print(network2.cells[0].layers[0])

tuple_index = convert_str_to_op_indices(param_list)
print("tuple_index:", tuple_index)