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
        
        # Register buffers - FIXED: only register freqs_cis once
        self.register_buffer('position_counter', torch.zeros(1, dtype=torch.int), persistent=False)
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(mla_module.model_args),
            persistent=False
        )
        
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
            # FIXED: Handle batched position_ids correctly
            if position_ids.dim() > 1:
                # For MLA.apply_rotary_emb to work, freqs_cis must be [seq_len, head_dim/2]
                # When position_ids has batch dimension, use only the first batch's positions
                position_ids_flat = position_ids[0]
                freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids_flat)
            else:
                # Non-batched position_ids (1D tensor)
                freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids)
        else:
            # No position_ids, use sequential positions
            freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len]
        
        # Ensure freqs_cis is on the right device
        freqs_cis = freqs_cis.to(device=device)
        
        # Call MLA forward
        try:
            output = self.mla(
                x=hidden_states,
                start_pos=start_pos,
                freqs_cis=freqs_cis,
                mask=mla_mask
            )
        except Exception as e:
            # Add debugging information if the forward call fails
            import logging
            logger = logging.getLogger(__name__)
            logger.error("--- Error during self.mla forward call ---", exc_info=True)
            logger.error(f"  hidden_states shape: {hidden_states.shape}")
            logger.error(f"  start_pos: {start_pos}")
            logger.error(f"  freqs_slice shape: {freqs_cis.shape if freqs_cis is not None else 'None'}")
            logger.error(f"  mla_mask shape: {mla_mask.shape if mla_mask is not None else 'None'}")
            raise e
        
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
}

transform_func_map = {
    "mla": transform_llama2_to_mla,
}

wrapper_map = {
    "mla": MLAAttentionWrapper,
}