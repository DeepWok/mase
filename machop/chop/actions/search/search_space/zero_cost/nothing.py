# 'skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'
import itertools
from xautodl.models import get_cell_based_tiny_net
from xautodl.models.cell_searchs.genotypes import Structure

def generate_arch_str(op_combinations):
    op_mapping = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
    arch_parts = []
    for i, ops in enumerate(op_combinations):
        # 生成当前层的操作字符串，并转换操作码为名称
        layer_str = '|'.join(f"{op_mapping[op]}~{i}" for op in ops)
        # 添加层分隔符
        if i > 0:
            layer_str = '|' + layer_str +'|' 
        arch_parts.append(layer_str)
    # 将加号前添加竖线，并去除"0~0+"部分
    arch_str = '+'.join(arch_parts).replace('skip_connect~0+', '')
    return arch_str


def generate_configs(config_dict):
    op_keys = sorted([key for key in config_dict['nas_zero_cost']['config'] if key.startswith('op_')], key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
    op_values = [config_dict['nas_zero_cost']['config'][key] for key in op_keys]
    
    # 按层分组操作
    layers = {}
    for key, values in zip(op_keys, op_values):
        layer_index = int(key.split('_')[1])
        if layer_index not in layers:
            layers[layer_index] = []
        layers[layer_index].append(values)

    layer_combinations = [list(itertools.product(*layer_ops)) for layer_ops in layers.values()]

    all_combinations = list(itertools.product(*layer_combinations))
    
    # 生成配置
    configs = []
    for combination in all_combinations:
        genotype = []
        for i, layer_ops in enumerate(combination):
            for op_idx, op_name in enumerate(layer_ops):
                if op_name != 'none':  # Exclude 'none' operations
                    genotype.append([(op_name, op_idx)])
        
        structure = Structure.gen_all(genotype, len(genotype) + 1, False)
        for struct in structure:
            if struct.check_valid():
                config = {
                    'name': config_dict['nas_zero_cost']['config']['name'][0],
                    'C': config_dict['nas_zero_cost']['config']['C'][0],
                    'N': config_dict['nas_zero_cost']['config']['N'][0],
                    'arch_str': struct.tostr(),
                    'num_classes': config_dict['nas_zero_cost']['config']['number_classes'][0]
                }
                configs.append(config)
                print("append")
    return configs



# 示例使用
config_dict = {
    'name': 'graph/software/zero_cost',
    'setup': {'by': 'name'},
    'seed': {'default': {'config': {'name': [None]}}},
    'nas_zero_cost': {
        'config': {
            'name': ['infer.tiny'],
            'C': [16],
            'N': [5],
            'op_0_0': [0],
            'op_1_0': [0, 1, 2],
            'op_2_1': [0, 1, 2],
            'op_2_2': [0, 1, 2],
            'op_3_1': [0, 1, 2],
            'op_3_2': [0, 1, 2],
            'op_3_3': [0, 1, 2],
            'number_classes': [10]
        }
    }
}

# 示例使用
configs = generate_configs(config_dict)
for config in configs[:10]:  # 打印前5个配置进行检查
    print(config)
    
def build_search_space(self, config_all):
    # choises = {}
    
    self.choices_flattened = generate_configs(config_all)
    
    self.choice_lengths_flattened = {
        k: len(v) for k, v in self.choices_flattened.items()
    }

network1 = get_cell_based_tiny_net(configs[10])
print(network1)