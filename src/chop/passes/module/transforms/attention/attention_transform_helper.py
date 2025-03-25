import torch
from typing import Optional
import math
from typing import Optional, Tuple, Union
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from chop.nn.attention.modules.mla import (
    ModelArgs, 
    MLA,
    RMSNorm
)
from chop.nn.attention.modules.mgqa import (
    MGQALayers,
)
from ...module_modify_helper import (
    get_module_by_name, 
    set_module_by_name,
)
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertSelfAttention, 
    BertSdpaSelfAttention,
    BertSelfOutput,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2SdpaAttention,
    GPT2Block,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer
)

def instantiate_attention_module(module, postfix, module_map, additional_module_args):
    # sdpa_attn = module.self
    # self_output = module.output
    additional_module_args = additional_module_args["config"]
    init_func = init_func_map[postfix]
    
    attention_module = init_func(
        module,
        config=additional_module_args,
    )

    return attention_module

def replace_attention_by_name(network, name, module, postfix):
    
    original = get_module_by_name(network, name)

    transform_func = transform_func_map[postfix]
    wrapper_class = wrapper_map[postfix]

    new = transform_func(original, module)
    wapper = wrapper_class(new)

    network = set_module_by_name(network, name, wapper)
    return network

def gpt2sdpa_to_mla_init(
    gpt2_block: GPT2Block,  
    config: dict 
) -> MLA:
    """
    Initialize and return an MLA module based on dimensions
    extracted from a GPT2SdpaAttention (within GPT2Block).
    
    Args:
        gpt2_block (GPT2Block): A GPT-2 block containing GPT2SdpaAttention as `.attn`.
        config (dict): A user config dict, which can contain nested "config" entries 
                       for MLA's ModelArgs.
                       e.g. {"config": {"max_batch_size": 8, "q_lora_rank": 0, ...}}
    Returns:
        MLA: A newly constructed MLA module with random initialization.
    """

    # GPT2Block -> GPT2SdpaAttention
    gpt2_sdpa_attn: GPT2SdpaAttention = gpt2_block.attn

    # gather GPT-2 attention hyperparams
    hidden_size = gpt2_sdpa_attn.embed_dim    # e.g., 768
    n_heads     = gpt2_sdpa_attn.num_heads    # e.g., 12
    head_dim = hidden_size // n_heads 

    # optional user config
    user_config = config.get("config", {})

    # Create ModelArgs for MLA
    model_args = ModelArgs(
        dim=hidden_size,
        n_heads=n_heads,
        qk_nope_head_dim=head_dim,  # Use full head dimension
        qk_rope_head_dim=0,          # Disable rotary embeddings
        v_head_dim=head_dim,
        max_seq_len=512,             # Match tokenizer max_length
        kv_lora_rank=512             # Increased rank for full sequences
    )

    



    # Construct MLA with those arguments
    mla_module = MLA(model_args)

    # Return the newly constructed module (randomly initialized)
    return mla_module


def gpt2sdpa_to_mgqa_init(
    gpt2_block: GPT2Block,  
    config: dict                       
) -> MGQALayers:

    layernorm1 = gpt2_block.ln_1
    gpt2_sdpa_attn  = gpt2_block.attn   # GPT2SdpaAttention
    layernorm2 = gpt2_block.ln_2
    gpt2_mlp   = gpt2_block.mlp    # GPT2MLP

    # Basic info from gpt2_sdpa_attn
    hidden_size = gpt2_sdpa_attn.embed_dim
    num_heads = gpt2_sdpa_attn.num_heads
    attn_drop = gpt2_sdpa_attn.attn_dropout.p

    ff_dropout_p = gpt2_mlp.dropout.p

    kv_heads = config.get("kv_heads", num_heads)
    kv_heads = num_heads // math.ceil(num_heads/ kv_heads)
    
    mgqa_kwargs = {
        "dim":      hidden_size,
        "heads":    num_heads, # number of query
        "kv_heads": kv_heads, # number of kv heads
        "one_kv_head":  config.get("one_kv_head", False), # force kv_heads to 1
        "causal":   True,
        "depth":    config.get("depth", 1),
        "dropout":  config.get("dropout", attn_drop),
        "flash":    config.get("flash", False),
        "talking_heads":config.get("talking_heads", False),
        "head_scale":   config.get("head_scale", False),
        "qk_norm":  config.get("qk_norm", False),
        "zero_init_output": config.get("zero_init_output", False),
        "shared_kv": config.get("shared_kv", False),
        "ff_dropout": ff_dropout_p,
        "pre_norm": True, 
        "resi_dual": False,
    }
    mgqa_layers = MGQALayers(**mgqa_kwargs)
    # mgqa automatically add a layernorm at the end, while gpt2 have layernorm 
    # at the end already
    mgqa_layers.final_norm = torch.nn.Identity() 
    return mgqa_layers


def transform_gpt2sdpa_to_mla(
    gpt2_block: GPT2Block,
    mla_attn: "MLA",
):
    gpt2_sdpa_attn: GPT2SdpaAttention = gpt2_block.attn
    embed_dim = gpt2_sdpa_attn.embed_dim

    target_dtype = mla_attn.wq.weight.dtype
    c_attn_weight = gpt2_sdpa_attn.c_attn.weight.to(target_dtype)
    c_attn_bias = gpt2_sdpa_attn.c_attn.bias.to(target_dtype) if gpt2_sdpa_attn.c_attn.bias is not None else None

    q_weight, k_weight, v_weight = torch.split(c_attn_weight, embed_dim, dim=1)
    if c_attn_bias is not None:
        q_bias, k_bias, v_bias = torch.split(c_attn_bias, embed_dim, dim=0)
    else:
        q_bias = k_bias = v_bias = None

    with torch.no_grad():
        mla_attn.wq.weight.copy_(q_weight)
        if mla_attn.wq.bias is None and q_bias is not None:
            mla_attn.wq.bias = torch.nn.Parameter(q_bias.clone())

    kv_weight = torch.cat([k_weight, v_weight], dim=0).to(torch.float32)

    # Compute SVD Rank Dynamically
    U, S, Vh = torch.linalg.svd(kv_weight, full_matrices=False)
    # explained_variance = (S**2) / (S**2).sum()
    # cumulative_variance = torch.cumsum(explained_variance, dim=0)
    rank = mla_attn.kv_lora_rank  # Use predefined rank from ModelArgs
    U_approx = U[:, :rank]
    S_approx = S[:rank]
    Vh_approx = Vh[:rank, :]

    # Ensure exact shape matching
    A = U_approx @ torch.diag(S_approx)
    B = Vh_approx

    # Pad/crop matrices to match MLA dimensions
    A = torch.nn.functional.pad(A, (0, mla_attn.wkv_b.weight.shape[1] - A.shape[1]))
    B = torch.nn.functional.pad(B, (0, mla_attn.wkv_a.weight.shape[1] - B.shape[1]))

    # Ensure shapes match expected MLA dimensions
    if A.shape != mla_attn.wkv_b.weight.shape:
        A = A[: mla_attn.wkv_b.weight.shape[0], : mla_attn.wkv_b.weight.shape[1]]

    if B.shape != mla_attn.wkv_a.weight.shape:
        B = B[: mla_attn.wkv_a.weight.shape[0], : mla_attn.wkv_a.weight.shape[1]]

    # Debug prints (optional)
    print(f"Adjusted A.shape: {A.shape}, Target shape: {mla_attn.wkv_b.weight.shape}")
    print(f"Adjusted B.shape: {B.shape}, Target shape: {mla_attn.wkv_a.weight.shape}")


    with torch.no_grad():
        mla_attn.wkv_b.weight.copy_(A.to(target_dtype))
        mla_attn.wkv_a.weight.copy_(B.to(target_dtype))

    # Adjust kv_norm
    # Adjust kv_norm
    if hasattr(mla_attn, "kv_norm") and mla_attn.kv_norm is not None:
        with torch.no_grad():
            # Ensure kv_norm is filled with a scalar for all dimensions
            kv_norm_fill_value = 0.9 + 0.1 * torch.rand_like(mla_attn.kv_norm.weight)
            # If kv_norm.weight is multi-dimensional, you might want to fill each dimension
            mla_attn.kv_norm.weight.data.copy_(kv_norm_fill_value)


    c_proj_weight = gpt2_sdpa_attn.c_proj.weight.to(target_dtype)
    c_proj_bias = gpt2_sdpa_attn.c_proj.bias.to(target_dtype) if gpt2_sdpa_attn.c_proj.bias is not None else None

    with torch.no_grad():
        mla_attn.wo.weight.copy_(c_proj_weight)
        if mla_attn.wo.bias is None and c_proj_bias is not None:
            mla_attn.wo.bias = torch.nn.Parameter(c_proj_bias.clone())

    return mla_attn


def transform_gpt2sdpa_to_mgqa(
    gpt2_block: GPT2Block,
    mgqa: MGQALayers,
):
    """
    Load weights and related info from a single GPT2SdpaAttention into the first MGQA attention layer.
    Assumes MGQALayers was set up with depth=1 or has at least one 'a' (attention) block.
    """
    layernorm1 = gpt2_block.ln_1
    gpt2sdpa  = gpt2_block.attn   # GPT2SdpaAttention
    layernorm2 = gpt2_block.ln_2
    gpt2_mlp   = gpt2_block.mlp    # GPT2MLP
    
    # 1) Retrieve the MGQA attention module
    mgqa_norms, mgqa_block, mgqa_residual = mgqa.layers[0]
    mgqa_attn = mgqa_block

    # 2) GPT2SdpaAttention: gather shapes & param references
    embed_dim = gpt2sdpa.embed_dim
    c_attn_weight = gpt2sdpa.c_attn.weight       # shape [embed_dim, 3*embed_dim]
    c_attn_bias   = gpt2sdpa.c_attn.bias         # shape [3*embed_dim]
    c_proj_weight = gpt2sdpa.c_proj.weight       # shape [embed_dim, embed_dim]
    c_proj_bias   = gpt2sdpa.c_proj.bias         # shape [embed_dim]

    # 3) Split out the Q/K/V weights from c_attn

    with torch.no_grad():
        q_w, k_w, v_w = torch.split(c_attn_weight, embed_dim, dim=1)
        mgqa_attn.to_q.weight.copy_(q_w) # query size remain identical, no change needed

        heads = mgqa_attn.heads
        kv_heads = mgqa_attn.kv_heads
        dim_head = embed_dim // heads

        if kv_heads < heads:
            k_r = k_w.reshape(embed_dim, heads, dim_head)
            v_r = v_w.reshape(embed_dim, heads, dim_head)
            g_size = heads // kv_heads
            k_r = k_r.reshape(embed_dim, kv_heads, g_size, dim_head).mean(dim=2)
            v_r = v_r.reshape(embed_dim, kv_heads, g_size, dim_head).mean(dim=2)
            k_w = k_r.reshape(embed_dim, kv_heads * dim_head)
            v_w = v_r.reshape(embed_dim, kv_heads * dim_head)
        
        if mgqa_attn.to_k is not None:
            mgqa_attn.to_k.weight.copy_(k_w)
        if mgqa_attn.to_v is not None:
            mgqa_attn.to_v.weight.copy_(v_w)

        # If your MGQA block has biases on Q/K/V, you could copy them here.
        # By default, from the MGQA code shown, bias=False, so we skip copying.

        # 4) Map c_proj -> mgqa_attn.to_out
        mgqa_attn.to_out.weight.copy_(c_proj_weight)
        if mgqa_attn.to_out.bias is not None:
            mgqa_attn.to_out.bias.copy_(c_proj_bias)

    mgqa_attn.attend.attn_dropout.p = gpt2sdpa.attn_dropout.p # Copy over dropout probability
    mgqa_attn.attend.causal = True  # GPT-2 standard

    return mgqa


# class MLAWrapper(torch.nn.Module):
#     def __init__(self, mla):
#         super().__init__()
#         self.mla = mla
#         # Initialize caches in BFloat16 upfront
#         self._init_caches()

#     def _init_caches(self):
#         """Ensure all caches start with BFloat16 dtype"""
#         for attr in ["kv_cache", "pe_cache", "k_cache", "v_cache"]:
#             if hasattr(self.mla, attr):
#                 cache = getattr(self.mla, attr)
#                 if cache is not None and cache.dtype != torch.bfloat16:
#                     setattr(self.mla, attr, cache.to(torch.bfloat16))

#     def forward(self, hidden_states, attention_mask=None, **kwargs):
#         # Convert inputs to BFloat16
#         if hidden_states is not None:
#             hidden_states = hidden_states.to(torch.bfloat16)
#             bsz, seqlen, _ = hidden_states.shape

#         # Process attention mask
#         if attention_mask is not None:
#             if attention_mask.dim() == 2:
#                 attention_mask = attention_mask[:, None, None, :]
#             attention_mask = attention_mask.expand(-1, self.mla.n_heads, -1, -1)
#             attention_mask = attention_mask.permute(0, 2, 1, 3)
#             attention_mask = attention_mask.to(torch.bfloat16)

#         # Enforce cache dtype before forward pass
#         self._init_caches()
        
#         # Generate rotary embeddings
#         rope_dim = getattr(self.mla, "qk_rope_head_dim", 0)
#         freqs_cis = self._create_rotary_embeddings(seqlen, rope_dim, hidden_states.device)
        
#         # Forward pass
#         output = self.mla(
#             x=hidden_states,
#             start_pos=0,
#             freqs_cis=freqs_cis,
#             mask=attention_mask,
#         )
        
#         return (output.to(torch.float32), None, None)

#     def _create_rotary_embeddings(self, seqlen, rope_dim, device):
#         if rope_dim > 0:
#             dummy_freq = torch.zeros((seqlen, rope_dim//2), 
#                             dtype=torch.float32, device=device)
#             return torch.polar(torch.ones_like(dummy_freq), dummy_freq).to(torch.bfloat16)
#         return torch.zeros((seqlen, 0), dtype=torch.bfloat16, device=device)

class GPT_MLAWrapper(torch.nn.Module):
    def __init__(self, mla):
        super().__init__()
        self.mla = mla
        # Initialize caches in BFloat16 upfront
        self._init_caches()

    def _init_caches(self):
        """Ensure all caches start with BFloat16 dtype."""
        for attr in ["kv_cache", "pe_cache", "k_cache", "v_cache"]:
            if hasattr(self.mla, attr):
                cache = getattr(self.mla, attr)
                if cache is not None and cache.dtype != torch.bfloat16:
                    setattr(self.mla, attr, cache.to(torch.bfloat16))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Convert inputs to BFloat16
        if hidden_states is not None:
            hidden_states = hidden_states.to(torch.bfloat16)
            bsz, seqlen, _ = hidden_states.shape

        # Process attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Add head and sequence dimensions
                
                attention_mask = attention_mask[:, None, None, :]  # Shape: (bsz, 1, 1, seqlen)
            # Expand mask to match the number of attention heads
            # attention_mask shape [bsz, 1, seqlen, endpos]
            attention_mask = attention_mask.expand(-1, self.mla.n_heads, -1, -1)  # Shape: (bsz, n_heads, seqlen, endpos)
            # Permute mask to match the expected format for MLA
            attention_mask = attention_mask[:, :, 0, :] # Shape: (bsz, n_heads, endpos)
            attention_mask = attention_mask.to(torch.bfloat16)

        # Enforce cache dtype before forward pass
        self._init_caches()
        
        # Generate rotary embeddings
        rope_dim = getattr(self.mla, "qk_rope_head_dim", 0)
        freqs_cis = self._create_rotary_embeddings(seqlen, rope_dim, hidden_states.device)
        
        # Forward pass
        output = self.mla(
            x=hidden_states,
            start_pos=0,
            freqs_cis=freqs_cis,
            mask=attention_mask,
        )
        
        # Convert output to Float32 for downstream compatibility
        return (output.to(torch.float32), None, None)

    def _create_rotary_embeddings(self, seqlen, rope_dim, device):
        """Generate rotary embeddings for the given sequence length and dimension."""
        if rope_dim > 0:
            # Create dummy frequencies for rotary embeddings
            dummy_freq = torch.zeros((seqlen, rope_dim // 2), 
                          dtype=torch.float32, device=device)
            # Convert to polar form (complex numbers) and cast to BFloat16
            return torch.polar(torch.ones_like(dummy_freq), dummy_freq).to(torch.bfloat16)
        # Return empty tensor if no rotary embeddings are needed
        return torch.zeros((seqlen, 0), dtype=torch.bfloat16, device=device)

class MGQAWrapper(torch.nn.Module):

    def __init__(self, mgqa):
        super().__init__()
        self.mgqa = mgqa

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        attention_mask = (attention_mask[:, 0, 0, :] == 0).bool()
        # Call MGQA with the relevant tensors.
        # Pass cross-attention if encoder_hidden_states is given, otherwise self-attention.
        if encoder_hidden_states is not None:
            out = self.mgqa(
                x=hidden_states,
                context=encoder_hidden_states,
                mask=attention_mask,
                context_mask=encoder_attention_mask,
            )
        else:
            out = self.mgqa(
                x=hidden_states,
                mask=attention_mask,
            )

        return (out, None, None)
    



def llama2_to_mla_init(
    module,  
    config: dict 
) -> MLA:
    """
    Initialize and return an MLA module based on dimensions
    extracted from a Llama attention module.
    
    Args:
        module: Either a LlamaDecoderLayer or a LlamaSdpaAttention/LlamaAttention module
        config (dict): Configuration dictionary
    Returns:
        MLA: A newly constructed MLA module with random initialization (not wrapped)
    """
    # Determine if we're dealing with a decoder layer or directly with an attention module
    if hasattr(module, 'self_attn'):
        # This is a LlamaDecoderLayer
        llama_attention = module.self_attn
    else:
        # This is already an attention module
        llama_attention = module

    # Extract parameters from the attention module
    if hasattr(llama_attention, 'hidden_size'):
        hidden_size = llama_attention.hidden_size
    elif hasattr(llama_attention, 'embed_dim'):
        hidden_size = llama_attention.embed_dim
    else:
        hidden_size = llama_attention.q_proj.weight.shape[1]
        
    if hasattr(llama_attention, 'num_heads'):
        n_heads = llama_attention.num_heads
    elif hasattr(llama_attention, 'num_attention_heads'):
        n_heads = llama_attention.num_attention_heads
    else:
        head_dim = getattr(llama_attention, 'head_dim', 64)
        n_heads = hidden_size // head_dim
        
    head_dim = hidden_size // n_heads

    print(f"Extracted dimensions: hidden_size={hidden_size}, n_heads={n_heads}, head_dim={head_dim}")

    # Optional user config
    user_config = config.get("config", {})

    # Create ModelArgs for MLA
    model_args = ModelArgs(
        dim=hidden_size,
        n_heads=n_heads,
        qk_nope_head_dim=0,
        qk_rope_head_dim=head_dim,
        v_head_dim=head_dim,
        max_seq_len=user_config.get("max_seq_len", 4096),
        max_batch_size=user_config.get("max_batch_size", 4),
        kv_lora_rank=min(hidden_size, 384)
    )

    # Construct MLA with those arguments
    mla_module = MLA(model_args)
    
    # Store model_args with the module for later use
    mla_module.model_args = model_args

    print(f"Created MLA with dimensions:")
    print(f"  wq.weight: {mla_module.wq.weight.shape}")
    print(f"  wkv_a.weight: {mla_module.wkv_a.weight.shape}")
    print(f"  wkv_b.weight: {mla_module.wkv_b.weight.shape}")
    print(f"  wo.weight: {mla_module.wo.weight.shape}")

    # Return the unwrapped MLA module
    return mla_module


def transform_llama2_to_mla(
    module,
    mla_attn: MLA,
):
    """
    Transform weights from a Llama attention module to an MLA module.
    
    Args:
        module: Llama attention module
        mla_attn (MLA): Target MLA module to be transformed
        
    Returns:
        MLA: Transformed MLA module (not wrapped)
    """
    # Determine source module
    if hasattr(module, 'self_attn'):
        llama_attention = module.self_attn
    else:
        llama_attention = module
    
    # Extract weights
    q_proj_weight = llama_attention.q_proj.weight
    k_proj_weight = llama_attention.k_proj.weight
    v_proj_weight = llama_attention.v_proj.weight
    o_proj_weight = llama_attention.o_proj.weight
    
    # Get target dtype
    target_dtype = mla_attn.wq.weight.dtype
    
    # Copy query weights
    with torch.no_grad():
        mla_attn.wq.weight.copy_(q_proj_weight.to(target_dtype))
    
    print(f"MLA wkv_b.weight shape: {mla_attn.wkv_b.weight.shape}")
    print(f"MLA wkv_a.weight shape: {mla_attn.wkv_a.weight.shape}")
    
    # Concatenate k and v weights for low-rank decomposition
    kv_weight = torch.cat([k_proj_weight, v_proj_weight], dim=0).to(torch.float32)
    print(f"KV concatenated shape: {kv_weight.shape}")
    
    # Get target dimensions
    b_rows, b_cols = mla_attn.wkv_b.weight.shape
    a_rows, a_cols = mla_attn.wkv_a.weight.shape
    
    # Use proper rank
    rank = min(b_cols, min(kv_weight.shape))
    print(f"Using rank: {rank}")
    
    # Compute SVD for low-rank approximation
    try:
        U, S, Vh = torch.linalg.svd(kv_weight, full_matrices=False)
        print(f"SVD successful: U shape: {U.shape}, S shape: {S.shape}, Vh shape: {Vh.shape}")
        
        # Truncate to rank
        U_trunc = U[:, :rank]
        S_trunc = torch.sqrt(S[:rank])
        Vh_trunc = Vh[:rank, :]
        
        # Create scaled A and B matrices
        A = (U_trunc @ torch.diag(S_trunc)).to(torch.float32)
        B = (torch.diag(S_trunc) @ Vh_trunc).to(torch.float32)
        
        print(f"Created A shape: {A.shape}, B shape: {B.shape}")
        
        # Create properly sized matrices
        A_resized = torch.zeros((b_rows, b_cols), dtype=torch.float32, device=A.device)
        B_resized = torch.zeros((a_rows, a_cols), dtype=torch.float32, device=B.device)
        
        # Fill with values from A and B
        repeat_rows_a = (b_rows + A.shape[0] - 1) // A.shape[0]
        A_repeated = A.repeat(repeat_rows_a, 1)
        A_resized[:, :b_cols] = A_repeated[:b_rows, :b_cols]
        
        repeat_rows_b = (a_rows + B.shape[0] - 1) // B.shape[0]
        B_repeated = B.repeat(repeat_rows_b, 1)
        B_resized[:, :a_cols] = B_repeated[:a_rows, :a_cols]
        
        print(f"Resized A shape: {A_resized.shape}, Resized B shape: {B_resized.shape}")
        
        # Copy the factorized weights
        with torch.no_grad():
            mla_attn.wkv_b.weight.copy_(A_resized.to(target_dtype))
            mla_attn.wkv_a.weight.copy_(B_resized.to(target_dtype))
    
    except Exception as e:
        print(f"SVD failed: {e}. Falling back to random initialization.")
        with torch.no_grad():
            torch.nn.init.normal_(mla_attn.wkv_b.weight, std=0.02)
            torch.nn.init.normal_(mla_attn.wkv_a.weight, std=0.02)
    
    # Adjust kv_norm if it exists
    if hasattr(mla_attn, "kv_norm") and mla_attn.kv_norm is not None:
        with torch.no_grad():
            kv_norm_fill_value = 0.9 + 0.1 * torch.rand_like(mla_attn.kv_norm.weight)
            mla_attn.kv_norm.weight.data.copy_(kv_norm_fill_value)
    
    # Copy output projection weights
    with torch.no_grad():
        mla_attn.wo.weight.copy_(o_proj_weight.to(target_dtype))
    
    # Return the transformed MLA (unwrapped)
    return mla_attn

class MLAWrapper(torch.nn.Module):
    """
    Wrapper for MLA to match LlamaAttention interface.
    """
    def __init__(self, mla_module):
        super().__init__()
        self.mla = mla_module
        
        # Get dimensions from the mla module
        self.hidden_size = mla_module.dim
        self.num_heads = mla_module.n_heads
        
        # Precompute frequency table once for efficiency
        self.freqs_cis = precompute_freqs_cis(mla_module.model_args)
        
        # Position counter for incremental decoding
        self.register_buffer('position_counter', torch.zeros(1, dtype=torch.int), persistent=False)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        """
        Adapter between LlamaAttention interface and MLA interface.
        """
        batch_size, seq_len = hidden_states.size()[:2]
        
        # Get target dtype from MLA parameters
        param = next(self.mla.parameters(), None)
        target_dtype = param.dtype if param is not None else torch.bfloat16
        device = hidden_states.device
        
        # Convert inputs to the target dtype
        hidden_states = hidden_states.to(target_dtype)
        
        # Convert attention mask format if needed
        mla_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mla_mask = attention_mask.squeeze(1)
            else:
                mla_mask = attention_mask
            
            # Convert mask to same dtype if it has non-boolean values
            if mla_mask.dtype != torch.bool:
                mla_mask = mla_mask.to(target_dtype)
        
        # Get start position for incremental decoding
        start_pos = 0
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                start_pos = past_key_value[0].size(2)  # [bsz, num_heads, seq_len, head_dim]
            elif hasattr(self, 'position_counter'):
                start_pos = self.position_counter.item()
                self.position_counter += seq_len
        
        # Get appropriate freqs_cis slice
        freqs_cis = self.freqs_cis
        if position_ids is not None:
            freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids.view(-1))
        else:
            freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len]
        
        # Ensure freqs_cis is on the right device
        freqs_cis = freqs_cis.to(device=device)
        
        # Call MLA forward
        output = self.mla(
            x=hidden_states,
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            mask=mla_mask
        )
        
        # Convert output to match original dtype if needed
        orig_dtype = kwargs.get('input_dtype', hidden_states.dtype)
        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)
        
        # Prepare outputs in Llama format
        attn_weights = None
        
        present_key_value = None
        if use_cache:
            head_dim = self.hidden_size // self.num_heads
            dummy_key = torch.zeros(
                (batch_size, self.num_heads, seq_len, head_dim),
                device=device, dtype=orig_dtype
            )
            dummy_value = torch.zeros_like(dummy_key)
            present_key_value = (dummy_key, dummy_value)
        
        return output, attn_weights, present_key_value

# In attention_transform_helper.py
class MLAAttentionWrapper(torch.nn.Module):
    """
    Wrapper for MLA to match LlamaAttention interface.
    Naming includes 'MLA' and 'Attention' to be easily detectable.
    """
    def __init__(self, mla_module):
        super().__init__()
        self.mla = mla_module
        self.is_mla_wrapper = True  # Attribute flag for detection
        
        # Get dimensions from the mla module
        self.hidden_size = mla_module.dim
        self.num_heads = mla_module.n_heads
        
        # Precompute frequency table once for efficiency
        self.freqs_cis = precompute_freqs_cis(mla_module.model_args)
        
        # Position counter for incremental decoding
        self.register_buffer('position_counter', torch.zeros(1, dtype=torch.int), persistent=False)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        """
        Adapter between LlamaAttention interface and MLA interface.
        """
        batch_size, seq_len = hidden_states.size()[:2]
        
        # Get target dtype from MLA parameters
        param = next(self.mla.parameters(), None)
        target_dtype = param.dtype if param is not None else torch.bfloat16
        device = hidden_states.device
        
        # Convert inputs to the target dtype
        hidden_states = hidden_states.to(target_dtype)
        
        # Convert attention mask format if needed
        mla_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mla_mask = attention_mask.squeeze(1)
            else:
                mla_mask = attention_mask
            
            # Convert mask to same dtype if it has non-boolean values
            if mla_mask.dtype != torch.bool:
                mla_mask = mla_mask.to(target_dtype)
        
        # Get start position for incremental decoding
        start_pos = 0
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                start_pos = past_key_value[0].size(2)  # [bsz, num_heads, seq_len, head_dim]
            elif hasattr(self, 'position_counter'):
                start_pos = self.position_counter.item()
                self.position_counter += seq_len
        
        # Get appropriate freqs_cis slice
        freqs_cis = self.freqs_cis
        if position_ids is not None:
            freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids.view(-1))
        else:
            freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len]
        
        # Ensure freqs_cis is on the right device
        freqs_cis = freqs_cis.to(device=device)
        
        # Call MLA forward
        output = self.mla(
            x=hidden_states,
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            mask=mla_mask
        )
        
        # Convert output to match original dtype if needed
        orig_dtype = kwargs.get('input_dtype', hidden_states.dtype)
        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)
        
        # Prepare outputs in Llama format
        attn_weights = None
        
        present_key_value = None
        if use_cache:
            head_dim = self.hidden_size // self.num_heads
            dummy_key = torch.zeros(
                (batch_size, self.num_heads, seq_len, head_dim),
                device=device, dtype=orig_dtype
            )
            dummy_value = torch.zeros_like(dummy_key)
            present_key_value = (dummy_key, dummy_value)
        
        return output, attn_weights, present_key_value

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Create base frequencies (using float32 for precision)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    # Apply YaRN scaling if needed
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    
    # Convert to complex exponentials (stays in float32)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


init_func_map = {
    "mla": llama2_to_mla_init,
    "mgqa": gpt2sdpa_to_mgqa_init
}

transform_func_map = {
    "mla": transform_llama2_to_mla,
    "mgqa": transform_gpt2sdpa_to_mgqa,
}

wrapper_map = {
    "mla": MLAAttentionWrapper,
    "mgqa": MGQAWrapper,
}
