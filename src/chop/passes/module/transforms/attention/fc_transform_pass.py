import torch
import torch.nn as nn
from ...module_modify_helper import get_module_by_name, set_module_by_name
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=None):
        super().__init__()
        if rank is None:
            rank = min(in_features, out_features) // 4
        
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=True)
    
    def forward(self, x):
        return self.B(self.A(x))

def gpt2sdpa_to_fc_init(attn_module: GPT2SdpaAttention, config: dict) -> nn.Module:
    hidden_size = attn_module.embed_dim
    
    # get desired rank from config
    use_low_rank = config.get("low_rank", True) 
    rank = config.get("rank", hidden_size // 4)  # default to 1/4 of hidden size
    
    if use_low_rank:
        # create low-rank approximation
        fc_layer = LowRankLinear(hidden_size, hidden_size, rank)
        
        # initialize with SVD of original weights
        with torch.no_grad():
            weight = attn_module.c_proj.weight.T
            # Apply SVD
            try:
                U, S, V = torch.svd(weight)
                # keep only top 'rank' singular values
                U = U[:, :rank]
                S = S[:rank]
                V = V[:, :rank]
                
                # set weights of low-rank approximation
                fc_layer.A.weight.copy_(V.T * torch.sqrt(S))
                fc_layer.B.weight.copy_(U * torch.sqrt(S))
                
                if attn_module.c_proj.bias is not None:
                    fc_layer.B.bias.copy_(attn_module.c_proj.bias)
            except Exception as e:
                print(f"SVD failed: {e}. Falling back to random initialization.")
                # if SVD fails, initialize with random weights
                if attn_module.c_proj.bias is not None:
                    fc_layer.B.bias.copy_(attn_module.c_proj.bias)
    else:
        # regular FC layer
        fc_layer = nn.Linear(hidden_size, hidden_size)
        with torch.no_grad():
            fc_layer.weight.copy_(attn_module.c_proj.weight.T)
            if attn_module.c_proj.bias is not None:
                fc_layer.bias.copy_(attn_module.c_proj.bias)
    
    return fc_layer

def fc_transform_pass(network, module_name: str, config: dict):
    attn_module = get_module_by_name(network, module_name)
    
    # initialize the new FC layer with low-rank approximation
    fc_layer = gpt2sdpa_to_fc_init(attn_module, config)
    
    # directly replace the original projection layer with our new layer
    attn_module.c_proj = fc_layer
    
    network = set_module_by_name(network, module_name, attn_module)
    
    return network
