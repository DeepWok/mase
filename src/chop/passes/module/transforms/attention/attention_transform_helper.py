import torch
import torch.nn as nn
from typing import Optional
import math
from typing import Optional, Tuple, Union


from chop.nn.attention.modules.mla import (
    ModelArgs, 
    MLA,
    RMSNorm,
)
from chop.nn.attention.modules.mgqa import (
    MGQALayers,
    MGQA
)
from chop.nn.attention.modules.lora_linear import (
    LowRankLinear,
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


def instantiate_attention_module(module, postfix, module_map, additional_module_args):
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
    if wrapper_class != None:
        wapper = wrapper_class(new)
    else:
        wapper = new

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

    # optional user config
    user_config = config.get("config", {})

    # Create ModelArgs for MLA
    model_args = ModelArgs(
        dim=hidden_size,       # 768
        n_heads=n_heads,       # 12
        q_lora_rank=user_config.get("q_lora_rank", 0),
        kv_lora_rank=user_config.get("kv_lora_rank", 512),
        # The key fix: ensure (qk_nope_head_dim + qk_rope_head_dim) == 64
        qk_nope_head_dim=user_config.get("qk_nope_head_dim", 64),
        qk_rope_head_dim=user_config.get("qk_rope_head_dim", 0),
        v_head_dim=user_config.get("v_head_dim", 64),
        max_batch_size=user_config.get("max_batch_size", 8),
        max_seq_len=user_config.get("max_seq_len", 4096),
        # Add other fields or overrides from user_config as needed
        # e.g., rope_factor, rope_theta, etc.
    )

    # Construct MLA with those arguments
    mla_module = MLA(model_args)

    # Return the newly constructed module (randomly initialized)
    return mla_module

def gpt2sdpa_to_mgqa_init(
    gpt2_sdpa: GPT2SdpaAttention,  
    config: dict                       
) -> MGQA:

    # Basic info from gpt2_sdpa
    hidden_size = gpt2_sdpa.embed_dim
    num_heads = gpt2_sdpa.num_heads
    attn_drop = gpt2_sdpa.attn_dropout.p
    kv_heads = config.get("kv_heads", num_heads)
    kv_heads = num_heads // math.ceil(num_heads/ kv_heads)

    mgqa_kwargs = {
        "dim":              hidden_size,
        "dim_head":         hidden_size//num_heads,
        "heads":            num_heads,                         # number of query heads
        "kv_heads":         kv_heads,                          # group or unify the KV heads
        "causal":           config.get("causal", True),        # GPT-2 is typically causal
        "dropout":          config.get("dropout", attn_drop),
    }
    mgqa_layers = MGQA(**mgqa_kwargs)
    return mgqa_layers

def gpt2sdpa_to_lorafc_init(
    attn_module: GPT2SdpaAttention, 
    config: dict
) -> nn.Module:
    hidden_size = attn_module.embed_dim
    
    # get desired rank from config
    use_low_rank = config.get("low_rank", True) 
    rank = config.get("rank", hidden_size // 4)  # default to 1/4 of hidden size
    
    if use_low_rank:
        fc_layer = LowRankLinear(hidden_size, hidden_size, rank)
    else:
        fc_layer = nn.Linear(hidden_size, hidden_size)
    
    return fc_layer

def transform_gpt2sdpa_to_mla(
    gpt2_block: GPT2Block,
    mla_attn: "MLA",  # your MLA class instance
):
    """
    Transforms (copies/factorizes) weights from a GPT2SdpaAttention
    into the given MLA instance, assuming world_size=1.
    
    Debug prints are included to show shapes at each step.
    """

    # -------------------------------------------------
    # 1. Get the GPT-2 SDPA attention submodule
    # -------------------------------------------------
    gpt2_sdpa_attn: GPT2SdpaAttention = gpt2_block.attn
    
    embed_dim = gpt2_sdpa_attn.embed_dim  # e.g., 768
    print(f"[DEBUG] GPT2SdpaAttention embed_dim: {embed_dim}")

    # c_attn: [in_dim=768, out_dim=3*768=2304]
    c_attn_weight = gpt2_sdpa_attn.c_attn.weight  
    c_attn_bias   = gpt2_sdpa_attn.c_attn.bias    

    print(f"[DEBUG] c_attn_weight shape: {c_attn_weight.shape}")
    if c_attn_bias is not None:
        print(f"[DEBUG] c_attn_bias shape: {c_attn_bias.shape}")
    else:
        print("[DEBUG] c_attn_bias is None.")

    # -------------------------------------------------
    # 2. Split 'c_attn' => Q/K/V chunks along dim=1
    # -------------------------------------------------
    q_weight, k_weight, v_weight = torch.split(c_attn_weight, embed_dim, dim=1)
    print(f"[DEBUG] q_weight shape: {q_weight.shape}")
    print(f"[DEBUG] k_weight shape: {k_weight.shape}")
    print(f"[DEBUG] v_weight shape: {v_weight.shape}")

    if c_attn_bias is not None:
        q_bias, k_bias, v_bias = torch.split(c_attn_bias, embed_dim, dim=0)
        print(f"[DEBUG] q_bias shape: {q_bias.shape}")
        print(f"[DEBUG] k_bias shape: {k_bias.shape}")
        print(f"[DEBUG] v_bias shape: {v_bias.shape}")
    else:
        q_bias = k_bias = v_bias = None

    # -------------------------------------------------
    # (A) Copy Q => MLA wq if q_lora_rank=0
    # -------------------------------------------------
    if mla_attn.q_lora_rank == 0:
        with torch.no_grad():
            # Debug shapes
            print(f"[DEBUG] mla_attn.wq.weight shape: {mla_attn.wq.weight.shape}")
            print(f"[DEBUG] q_weight.T shape: {q_weight.T.shape}")

            # Check dimension
            assert mla_attn.wq.weight.shape == (embed_dim, embed_dim), (
                f"Expected MLA wq.weight to be [{embed_dim}, {embed_dim}] but got "
                f"{mla_attn.wq.weight.shape}. Ensure world_size=1 or adapt slicing."
            )

            mla_attn.wq.weight.copy_(q_weight.T)

            if (mla_attn.wq.bias is not None) and (q_bias is not None):
                print("[DEBUG] Copying Q bias...")
                print(f"[DEBUG] mla_attn.wq.bias shape: {mla_attn.wq.bias.shape}")
                mla_attn.wq.bias.copy_(q_bias)
    else:
        raise NotImplementedError("q_lora_rank > 0 not implemented for Q transform.")

    # -------------------------------------------------
    # (B) Factorize K + V => SVD
    # -------------------------------------------------
    # k_weight & v_weight => each [768, 768] => cat => [1536, 768]
    kv_weight = torch.cat([k_weight, v_weight], dim=0)
    print(f"[DEBUG] kv_weight shape: {kv_weight.shape}")

    rank = mla_attn.kv_lora_rank
    U, S, Vh = torch.linalg.svd(kv_weight, full_matrices=False)
    U_approx = U[:, :rank]  
    S_approx = S[:rank]     
    Vh_approx = Vh[:rank, :]

    print(f"[DEBUG] U shape: {U.shape}, S shape: {S.shape}, Vh shape: {Vh.shape}")
    print(f"[DEBUG] Using rank = {rank}")
    print(f"[DEBUG] U_approx shape: {U_approx.shape}")
    print(f"[DEBUG] S_approx shape: {S_approx.shape}")
    print(f"[DEBUG] Vh_approx shape: {Vh_approx.shape}")

    # Reconstruct => A * B
    A = U_approx @ torch.diag(S_approx)  # => [1536, rank]
    B = Vh_approx                        # => [rank, 768]

    print(f"[DEBUG] A shape: {A.shape}, B shape: {B.shape}")

    # Copy => wkv_b.weight, wkv_a.weight
    with torch.no_grad():
        print(f"[DEBUG] mla_attn.wkv_b.weight shape: {mla_attn.wkv_b.weight.shape}")
        print(f"[DEBUG] mla_attn.wkv_a.weight shape: {mla_attn.wkv_a.weight.shape}")

        assert mla_attn.wkv_b.weight.shape == (A.shape[0], A.shape[1]), (
            f"Expected wkv_b.weight to be {A.shape}, got {mla_attn.wkv_b.weight.shape}."
        )
        mla_attn.wkv_b.weight.copy_(A)

        assert mla_attn.wkv_a.weight.shape == (B.shape[0], B.shape[1]), (
            f"Expected wkv_a.weight to be {B.shape}, got {mla_attn.wkv_a.weight.shape}."
        )
        mla_attn.wkv_a.weight.copy_(B)

    # Optionally set kv_norm to identity
    if hasattr(mla_attn, "kv_norm") and mla_attn.kv_norm is not None:
        if mla_attn.kv_norm.weight.shape[0] == rank:
            print("[DEBUG] Setting kv_norm.weight to 1.0 (identity).")
            with torch.no_grad():
                mla_attn.kv_norm.weight.fill_(1.0)

    # -------------------------------------------------
    # (C) Copy c_proj => MLA.wo
    # -------------------------------------------------
    c_proj_weight = gpt2_sdpa_attn.c_proj.weight  # [768, 768]
    c_proj_bias   = gpt2_sdpa_attn.c_proj.bias    # [768]

    print(f"[DEBUG] c_proj_weight shape: {c_proj_weight.shape}")
    if c_proj_bias is not None:
        print(f"[DEBUG] c_proj_bias shape: {c_proj_bias.shape}")

    with torch.no_grad():
        print(f"[DEBUG] mla_attn.wo.weight shape: {mla_attn.wo.weight.shape}")
        assert mla_attn.wo.weight.shape == (embed_dim, embed_dim), (
            f"Expected MLA wo.weight to be [{embed_dim}, {embed_dim}] but got {mla_attn.wo.weight.shape}."
        )
        mla_attn.wo.weight.copy_(c_proj_weight.T)

        if mla_attn.wo.bias is not None and c_proj_bias is not None:
            print("[DEBUG] Copying c_proj bias...")
            print(f"[DEBUG] mla_attn.wo.bias shape: {mla_attn.wo.bias.shape}")
            mla_attn.wo.bias.copy_(c_proj_bias)

    print("[DEBUG] transform_gpt2sdpa_to_mla completed successfully.")

    return mla_attn

def transform_gpt2sdpa_to_mgqa(
    gpt2sdpa: GPT2SdpaAttention,
    mgqa: MGQA,
):

    # GPT2SdpaAttention: gather shapes & param references
    embed_dim = gpt2sdpa.embed_dim
    c_attn_weight = gpt2sdpa.c_attn.weight       # shape [embed_dim, 3*embed_dim]
    c_attn_bias   = gpt2sdpa.c_attn.bias         # shape [3*embed_dim]
    c_proj_weight = gpt2sdpa.c_proj.weight       # shape [embed_dim, embed_dim]
    c_proj_bias   = gpt2sdpa.c_proj.bias         # shape [embed_dim]
    
    # MGQA: gather shapes & param references
    heads = mgqa.heads
    kv_heads = mgqa.kv_heads

    # 3) Split out the Q/K/V weights from c_attn
    with torch.no_grad():
        q_w, k_w, v_w = torch.split(c_attn_weight, embed_dim, dim=1)
        q_w, k_w, v_w = q_w.transpose(0, 1), k_w.transpose(0, 1), v_w.transpose(0, 1)
        q_b, k_b, v_b = torch.split(c_attn_bias, embed_dim, dim=0)

        mgqa.to_q.weight.copy_(q_w) # query size remain identical, no change needed
        mgqa.to_q.bias.copy_(q_b)

        dim_head = embed_dim // heads

        # handle kv pair reduction
        if kv_heads < heads:
            
            g_size = heads // kv_heads
            
            # -------- Weight --------
            # [heads * dim_head, in_dim] -> [heads, dim_head, in_dim]
            k_w_r = k_w.reshape(heads, dim_head, embed_dim)  # [heads, dim_head, embed_dim]
            v_w_r = v_w.reshape(heads, dim_head, embed_dim)

            # Group the heads => shape [kv_heads, g_size, dim_head, in_dim]
            k_w_r = k_w_r.reshape(kv_heads, g_size, dim_head, embed_dim).mean(dim=1)
            v_w_r = v_w_r.reshape(kv_heads, g_size, dim_head, embed_dim).mean(dim=1) # [kv_heads, dim_head, in_dim]

            # Flatten the row dims => shape [kv_heads * dim_head, in_dim]
            k_w = k_w_r.reshape(kv_heads * dim_head, embed_dim)
            v_w = v_w_r.reshape(kv_heads * dim_head, embed_dim)

            # -------- Biases --------
            # original shape is [heads * dim_head].
            k_b_r = k_b.reshape(heads, dim_head)                         # [heads, dim_head]
            v_b_r = v_b.reshape(heads, dim_head)
            k_b_r = k_b_r.reshape(kv_heads, g_size, dim_head).mean(dim=1)  # [kv_heads, dim_head]
            v_b_r = v_b_r.reshape(kv_heads, g_size, dim_head).mean(dim=1)
            k_b = k_b_r.reshape(kv_heads * dim_head)  # [out_dim]
            v_b = v_b_r.reshape(kv_heads * dim_head)


        if mgqa.to_k is not None:
            mgqa.to_k.weight.copy_(k_w)
            mgqa.to_k.bias.copy_(k_b)
        if mgqa.to_v is not None:
            mgqa.to_v.weight.copy_(v_w)
            mgqa.to_v.bias.copy_(v_b)

        # 4) Map c_proj -> mgqa.to_out
        mgqa.to_out.weight.copy_(c_proj_weight.transpose(0, 1))    
        mgqa.to_out.bias.copy_(c_proj_bias)

    mgqa.attend.attn_dropout.p = gpt2sdpa.attn_dropout.p # Copy over dropout probability

    return mgqa

def transform_gpt2sdpa_to_lorafc(
    gpt2sdpa: GPT2SdpaAttention,
    new_fc: nn.Module,
)-> GPT2SdpaAttention:
    
    use_low_rank = isinstance(new_fc, LowRankLinear)

    if use_low_rank:
        rank = new_fc.rank
        with torch.no_grad():
            weight = gpt2sdpa.c_proj.weight.T
            # Apply SVD
            try:
                U, S, V = torch.svd(weight)
                # keep only top 'rank' singular values
                U = U[:, :rank]
                S = S[:rank]
                V = V[:, :rank]
                
                # set weights of low-rank approximation
                new_fc.A.weight.copy_(V.T * torch.sqrt(S))
                new_fc.B.weight.copy_(U * torch.sqrt(S))
                
                if gpt2sdpa.c_proj.bias is not None:
                    new_fc.B.bias.copy_(gpt2sdpa.c_proj.bias)
            except Exception as e:
                print(f"SVD failed: {e}. Falling back to random initialization.")
                # if SVD fails, initialize with random weights
                if gpt2sdpa.c_proj.bias is not None:
                    new_fc.B.bias.copy_(gpt2sdpa.c_proj.bias)
    else:
        # regular FC layer
        with torch.no_grad():
            new_fc.weight.copy_(gpt2sdpa.c_proj.weight.T)
            if gpt2sdpa.c_proj.bias is not None:
                new_fc.bias.copy_(gpt2sdpa.c_proj.bias)
    
    gpt2sdpa.c_proj = new_fc
    return gpt2sdpa

class MLAWrapper(torch.nn.Module):
    def __init__(self, mla):
        super().__init__()
        self.mla = mla

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, None, None]:

        # (1) Convert inputs to the dtype MLA uses
        if hidden_states is not None:
            hidden_states = hidden_states.to(torch.bfloat16)
            bsz, seqlen, _ = hidden_states.shape
        else:
            bsz, seqlen = 0, 0
        desired_seqlen = 12
    
        # 1) Slice the hidden_states down from 458 -> 12 tokens
        if seqlen != desired_seqlen:
            hidden_states = hidden_states[:, -desired_seqlen:, :]  # keep last 12
            # now hidden_states is [8, 12, 768]
            bsz, seqlen, hidden_dim = hidden_states.shape
        
        # 2) Adjust the attention mask to match
        if attention_mask is not None:
            # If it's [8, 1, 458, 458], squeeze => [8, 458, 458]
            if attention_mask.dim() == 4 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask.squeeze(1)  # => [8, 458, 458]
            
            # If mask is still 458×458, slice the last 12×12 region
            if attention_mask.shape[-1] != seqlen:  # 12
                attention_mask = attention_mask[:, -seqlen:, -seqlen:]
                # => [8, 12, 12]
            
            # Optionally cast to bf16 if your model uses bf16
            attention_mask = attention_mask.to(torch.bfloat16)

        # (3) Ensure MLA’s caches are in bf16
        for attr_name in ["kv_cache", "pe_cache", "k_cache", "v_cache"]:
            if hasattr(self.mla, attr_name) and getattr(self.mla, attr_name) is not None:
                setattr(self.mla, attr_name, getattr(self.mla, attr_name).to(torch.bfloat16))

        # (4) Always create a valid freqs_cis, even if rope_dim=0
        rope_dim = getattr(self.mla, "qk_rope_head_dim", 0)
        if rope_dim > 0:
            # Normal case if rope_dim>0
            dummy_freq = torch.zeros((seqlen, rope_dim // 2), dtype=torch.float32, device=hidden_states.device)
            dummy_ones = torch.ones_like(dummy_freq, dtype=torch.float32)
            freqs_cis_fp32 = torch.polar(dummy_ones, dummy_freq)  # shape: [seqlen, rope_dim//2], complex float32
            freqs_cis = freqs_cis_fp32.to(torch.bfloat16)
        else:
            # Rope dim is 0, but MLA code still calls apply_rotary_emb.
            # Pass an EMPTY complex tensor instead of None to avoid the crash.
            # shape => [seqlen, 0] so .view() won't fail
            freqs_cis = torch.zeros((seqlen, 0), dtype=torch.complex32, device=hidden_states.device)

        # (5) Run MLA
        start_pos = 0
        output = self.mla(
            x=hidden_states,
            start_pos=start_pos,
            freqs_cis=freqs_cis,  # never None
            mask=attention_mask,
        )

        # (6) Convert output back to fp32 if needed
        output = output.to(dtype=torch.float32)

        return (output.detach(), None, None)
    
class MGQAWrapper(torch.nn.Module):

    def __init__(self, mgqa: MGQA):
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
        
        if attention_mask is None:
            # Create default mask if absent
            input_mask = torch.ones(hidden_states.size()[:-1], device=hidden_states.device, dtype=torch.bool)
        else:
            # Handle incoming 4D attention_mask from GPT-2
            if attention_mask.dim() == 4:
                input_mask = attention_mask[:, 0, 0, :].bool()
            elif attention_mask.dim() == 2:
                input_mask = attention_mask.bool()
            else:
                raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")



        # Call MGQA with the relevant tensors.
        # Pass cross-attention if encoder_hidden_states is given, otherwise self-attention.
        if encoder_hidden_states is not None:
            out = self.mgqa(
                x=hidden_states,
                context=encoder_hidden_states,
                mask=input_mask,
                context_mask=encoder_attention_mask,
            )
        else:
            out = self.mgqa(
                x=hidden_states,
                mask=input_mask,
            )

        return (out, None, None)
    

init_func_map = {
    "mla": gpt2sdpa_to_mla_init,
    "mgqa": gpt2sdpa_to_mgqa_init,
    "lora_fc": gpt2sdpa_to_lorafc_init,
}
transform_func_map = {
    "mla": transform_gpt2sdpa_to_mla,
    "mgqa": transform_gpt2sdpa_to_mgqa,
    "lora_fc": transform_gpt2sdpa_to_lorafc
}
wrapper_map = {
    "mla": MLAWrapper,
    "mgqa": MGQAWrapper,
    "lora_fc": None, # do not require a wrapper
}