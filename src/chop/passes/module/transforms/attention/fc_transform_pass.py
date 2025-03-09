import torch
import torch.nn as nn
from ...module_modify_helper import get_module_by_name, set_module_by_name
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention

def gpt2sdpa_to_fc_init(attn_module: GPT2SdpaAttention, config: dict) -> nn.Module:
    # for GPT2SdpaAttention
    # single linear layer mapping hidden states
    # using the query portion of the original c_attn weights
    
    # use the embed_dim from the attention module
    hidden_size = attn_module.embed_dim  
    fc_layer = nn.Linear(hidden_size, hidden_size)
    with torch.no_grad():
        # take the first hidden_size columns as the query weight
        fc_layer.weight.copy_(attn_module.c_attn.weight[:, :hidden_size])
        if attn_module.c_attn.bias is not None:
            # copy the first hidden_size bias values
            fc_layer.bias.copy_(attn_module.c_attn.bias[:hidden_size])
    return fc_layer


def transform_gpt2sdpa_to_fc(attn_module: GPT2SdpaAttention, fc_module: nn.Module) -> nn.Module:
    # replace the attention block with FC
    # since simple linear mapping, no further parameter transformation needed

    return fc_module

class FCWrapper(nn.Module):
    # wrapper for the FC module to maintain the same interface as attention
    # forward() receives hidden_states and returns a tuple with the output and a placeholder for cached values
   
    def __init__(self, fc_module: nn.Module):
        super().__init__()
        self.fc = fc_module

    def forward(self, hidden_states, **kwargs):
        output = self.fc(hidden_states)
        # return a tuple with output and None (or an appropriate placeholder for present)
        return (output, None)

init_func_map = {
    "mla": None,   # 已有的
    "mgqa": None,  # 已有的
    "fc": gpt2sdpa_to_fc_init,
}

transform_func_map = {
    "mla": None,   # 已有的
    "mgqa": None,  # 已有的
    "fc": transform_gpt2sdpa_to_fc,
}

wrapper_map = {
    "mla": None,   # 已有的
    "mgqa": None,  # 已有的
    "fc": FCWrapper,
}

def fc_transform_pass(network, module_name: str, config: dict):
    # replace the attention module specified by 'module_name'
   
    original = get_module_by_name(network, module_name)         # 1.locate the original attention module
    fc_module = gpt2sdpa_to_fc_init(original, config)           # 2.initialize a FC module (from the original attention's parameters)
    transformed = transform_gpt2sdpa_to_fc(original, fc_module) # 3.perform any parameter transformation (if needed)
    wrapped = FCWrapper(transformed)                            # 4.wrap the new module
    network = set_module_by_name(network, module_name, wrapped) # 5.replace original module in the network
    return network
