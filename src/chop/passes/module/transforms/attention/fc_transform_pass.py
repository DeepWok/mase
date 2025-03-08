import torch
import torch.nn as nn

from ...module_modify_helper import get_module_by_name, set_module_by_name

class GatedFCWrapper(nn.Module):
    def __init__(self, fc_layer: nn.Linear, input_dim: int):
        super().__init__()
        self.fc = fc_layer
        self.gate = nn.Linear(input_dim, input_dim)
    
    def forward(self, hidden_states, **kwargs):
        fc_output = self.fc(hidden_states)
        gate_value = torch.sigmoid(self.gate(hidden_states))
        output = gate_value * fc_output + (1 - gate_value) * hidden_states
        return (output,)

def gpt2sdpa_to_gated_fc(attn_module, config: dict) -> nn.Module:
    # 假设 attn_module.self.query 存在并能获取 in_features
    hidden_size = attn_module.self.query.in_features
    fc_layer = nn.Linear(hidden_size, hidden_size)
    with torch.no_grad():
        fc_layer.weight.copy_(attn_module.self.query.weight.data)
        if attn_module.self.query.bias is not None and fc_layer.bias is not None:
            fc_layer.bias.copy_(attn_module.self.query.bias.data)
    
    return GatedFCWrapper(fc_layer, hidden_size)

def fc_transform_pass(network, module_name: str, config: dict):
    original = get_module_by_name(network, module_name)
    gated_fc = gpt2sdpa_to_gated_fc(original, config)
    network = set_module_by_name(network, module_name, gated_fc)
    return network
