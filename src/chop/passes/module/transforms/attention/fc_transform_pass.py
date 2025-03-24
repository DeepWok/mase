import torch
import torch.nn as nn
from ...module_modify_helper import get_module_by_name, set_module_by_name
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention

class ResidualFC(nn.Module):
    def __init__(self, new_fc: nn.Module, original_fc: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.new_fc = new_fc
        self.original_fc = original_fc
        self.alpha = alpha 

    def forward(self, hidden_states, **kwargs):
        new_out = self.new_fc(hidden_states)
        orig_out = self.original_fc(hidden_states)
        output = self.alpha * new_out + (1 - self.alpha) * orig_out
        return output


class ResidualFCWrapper(nn.Module):
    def __init__(self, new_fc: nn.Module, original_fc: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.new_fc = new_fc
        self.original_fc = original_fc
        self.alpha = alpha 

    def forward(self, hidden_states, **kwargs):
        new_out = self.new_fc(hidden_states)
        orig_out = self.original_fc(hidden_states)
        output = self.alpha * new_out + (1 - self.alpha) * orig_out
        return output
        
def gpt2sdpa_to_fc_init(attn_module: GPT2SdpaAttention, config: dict) -> nn.Module:
    # for GPT2SdpaAttention
    # single linear layer mapping hidden states
    # using the query portion of the original c_attn weights
    
    # use the embed_dim from the attention module
    hidden_size = attn_module.embed_dim
    fc_layer = nn.Linear(hidden_size, hidden_size)
    with torch.no_grad():
        fc_layer.weight.copy_(attn_module.c_proj.weight)
        if attn_module.c_proj.bias is not None:
            fc_layer.bias.copy_(attn_module.c_proj.bias)
    return fc_layer

def fc_transform_pass(network, module_name: str, config: dict):
    attn_module = get_module_by_name(network, module_name)
    fc_layer = gpt2sdpa_to_fc_init(attn_module, config)
    original_fc = attn_module.c_proj
    alpha = config.get("alpha", 0.6)
    wrapper = ResidualFCWrapper(fc_layer, original_fc, alpha)
    attn_module.c_proj = wrapper
    network = set_module_by_name(network, module_name, attn_module)
    return network
    
