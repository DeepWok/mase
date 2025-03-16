import torch
import torch.nn as nn
from ...module_modify_helper import get_module_by_name, set_module_by_name
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention

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


# def transform_gpt2sdpa_to_fc(attn_module: GPT2SdpaAttention, fc_module: nn.Module) -> nn.Module:
#     # replace the attention block with FC
#     # since simple linear mapping, no further parameter transformation needed

#     return fc_module

# class FCWrapper(nn.Module):
#     # wrapper for the FC module to maintain the same interface as attention
#     # forward() receives hidden_states and returns a tuple with the output and a placeholder for cached values
   
#     def __init__(self, fc_module: nn.Module):
#         super().__init__()
#         self.fc = fc_module

#     def forward(self, hidden_states, **kwargs):
#         output = self.fc(hidden_states)
#         # return a tuple with output and None (or an appropriate placeholder for present)
#         return (output, None)

# init_func_map = {
#     "mla": None,   # 已有的
#     "mgqa": None,  # 已有的
#     "fc": gpt2sdpa_to_fc_init,
# }

# transform_func_map = {
#     "mla": None,   # 已有的
#     "mgqa": None,  # 已有的
#     "fc": transform_gpt2sdpa_to_fc,
# }

# wrapper_map = {
#     "mla": None,   # 已有的
#     "mgqa": None,  # 已有的
#     "fc": FCWrapper,
# }

def fc_transform_pass(network, module_name: str, config: dict):
    attn_module = get_module_by_name(network, module_name)
    fc_layer = gpt2sdpa_to_fc_init(attn_module, config)
    original_fc = attn_module.c_proj
    alpha = config.get("alpha", 0.6)
    wrapper = ResidualFCWrapper(fc_layer, original_fc, alpha)
    attn_module.c_proj = wrapper
    network = set_module_by_name(network, module_name, attn_module)
    return network
    
# def fc_transform_pass(network, module_name: str, config: dict):
#     # replace the attention module specified by 'module_name'
   
#     original = get_module_by_name(network, module_name)         # 1.locate the original attention module
#     fc_module = gpt2sdpa_to_fc_init(original, config)           # 2.initialize a FC module (from the original attention's parameters)
#     transformed = transform_gpt2sdpa_to_fc(original, fc_module) # 3.perform any parameter transformation (if needed)
#     wrapped = FCWrapper(transformed)                            # 4.wrap the new module
#     network = set_module_by_name(network, module_name, wrapped) # 5.replace original module in the network
#     return network
