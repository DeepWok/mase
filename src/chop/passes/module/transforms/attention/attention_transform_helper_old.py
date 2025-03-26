import torch
from typing import Optional
import math
from typing import Optional, Tuple, Union

from chop.nn.attention.modules.mla import ModelArgs, MLA, RMSNorm
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


def gpt2sdpa_to_mla_init(gpt2_block: GPT2Block, config: dict) -> MLA:
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
    hidden_size = gpt2_sdpa_attn.embed_dim  # e.g., 768
    n_heads = gpt2_sdpa_attn.num_heads  # e.g., 12

    # optional user config
    user_config = config.get("config", {})

    # Create ModelArgs for MLA
    model_args = ModelArgs(
        dim=hidden_size,  # 768
        n_heads=n_heads,  # 12
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


def gpt2sdpa_to_mgqa_init(gpt2_block: GPT2Block, config: dict) -> MGQALayers:

    layernorm1 = gpt2_block.ln_1
    gpt2_sdpa_attn = gpt2_block.attn  # GPT2SdpaAttention
    layernorm2 = gpt2_block.ln_2
    gpt2_mlp = gpt2_block.mlp  # GPT2MLP

    # Basic info from gpt2_sdpa_attn
    hidden_size = gpt2_sdpa_attn.embed_dim
    num_heads = gpt2_sdpa_attn.num_heads
    attn_drop = gpt2_sdpa_attn.attn_dropout.p

    ff_dropout_p = gpt2_mlp.dropout.p

    kv_heads = config.get("kv_heads", num_heads)
    kv_heads = num_heads // math.ceil(num_heads / kv_heads)

    mgqa_kwargs = {
        "dim": hidden_size,
        "heads": num_heads,  # number of query
        "kv_heads": kv_heads,  # number of kv heads
        "one_kv_head": config.get("one_kv_head", False),  # force kv_heads to 1
        "causal": True,
        "depth": config.get("depth", 1),
        "dropout": config.get("dropout", attn_drop),
        "flash": config.get("flash", False),
        "talking_heads": config.get("talking_heads", False),
        "head_scale": config.get("head_scale", False),
        "qk_norm": config.get("qk_norm", False),
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


# def transform_gpt2sdpa_to_mla(
#     gpt2_block: GPT2Block,
#     mla_attn: "MLA",  # your MLA class instance
# ):
#     """
#     Transforms (copies/factorizes) weights from a GPT2SdpaAttention
#     into the given MLA instance, assuming world_size=1.

#     Debug prints are included to show shapes at each step.
#     """

#     # -------------------------------------------------
#     # 1. Get the GPT-2 SDPA attention submodule
#     # -------------------------------------------------
#     gpt2_sdpa_attn: GPT2SdpaAttention = gpt2_block.attn

#     embed_dim = gpt2_sdpa_attn.embed_dim  # e.g., 768
#     print(f"[DEBUG] GPT2SdpaAttention embed_dim: {embed_dim}")

#     # c_attn: [in_dim=768, out_dim=3*768=2304]
#     c_attn_weight = gpt2_sdpa_attn.c_attn.weight
#     c_attn_bias   = gpt2_sdpa_attn.c_attn.bias

#     print(f"[DEBUG] c_attn_weight shape: {c_attn_weight.shape}")
#     if c_attn_bias is not None:
#         print(f"[DEBUG] c_attn_bias shape: {c_attn_bias.shape}")
#     else:
#         print("[DEBUG] c_attn_bias is None.")

#     # -------------------------------------------------
#     # 2. Split 'c_attn' => Q/K/V chunks along dim=1
#     # -------------------------------------------------
#     q_weight, k_weight, v_weight = torch.split(c_attn_weight, embed_dim, dim=1)
#     print(f"[DEBUG] q_weight shape: {q_weight.shape}")
#     print(f"[DEBUG] k_weight shape: {k_weight.shape}")
#     print(f"[DEBUG] v_weight shape: {v_weight.shape}")

#     if c_attn_bias is not None:
#         q_bias, k_bias, v_bias = torch.split(c_attn_bias, embed_dim, dim=0)
#         print(f"[DEBUG] q_bias shape: {q_bias.shape}")
#         print(f"[DEBUG] k_bias shape: {k_bias.shape}")
#         print(f"[DEBUG] v_bias shape: {v_bias.shape}")
#     else:
#         q_bias = k_bias = v_bias = None

#     # -------------------------------------------------
#     # (A) Copy Q => MLA wq if q_lora_rank=0
#     # -------------------------------------------------
#     if mla_attn.q_lora_rank == 0:
#         with torch.no_grad():
#             # Debug shapes
#             print(f"[DEBUG] mla_attn.wq.weight shape: {mla_attn.wq.weight.shape}")
#             print(f"[DEBUG] q_weight.T shape: {q_weight.T.shape}")

#             # Check dimension
#             assert mla_attn.wq.weight.shape == (embed_dim, embed_dim), (
#                 f"Expected MLA wq.weight to be [{embed_dim}, {embed_dim}] but got "
#                 f"{mla_attn.wq.weight.shape}. Ensure world_size=1 or adapt slicing."
#             )

#             mla_attn.wq.weight.copy_(q_weight.T)

#             if (mla_attn.wq.bias is not None) and (q_bias is not None):
#                 print("[DEBUG] Copying Q bias...")
#                 print(f"[DEBUG] mla_attn.wq.bias shape: {mla_attn.wq.bias.shape}")
#                 mla_attn.wq.bias.copy_(q_bias)
#     else:
#         raise NotImplementedError("q_lora_rank > 0 not implemented for Q transform.")

#     # -------------------------------------------------
#     # (B) Factorize K + V => SVD
#     # -------------------------------------------------
#     # k_weight & v_weight => each [768, 768] => cat => [1536, 768]
#     kv_weight = torch.cat([k_weight, v_weight], dim=0)
#     print(f"[DEBUG] kv_weight shape: {kv_weight.shape}")

#     rank = mla_attn.kv_lora_rank
#     U, S, Vh = torch.linalg.svd(kv_weight, full_matrices=False)
#     U_approx = U[:, :rank]
#     S_approx = S[:rank]
#     Vh_approx = Vh[:rank, :]

#     print(f"[DEBUG] U shape: {U.shape}, S shape: {S.shape}, Vh shape: {Vh.shape}")
#     print(f"[DEBUG] Using rank = {rank}")
#     print(f"[DEBUG] U_approx shape: {U_approx.shape}")
#     print(f"[DEBUG] S_approx shape: {S_approx.shape}")
#     print(f"[DEBUG] Vh_approx shape: {Vh_approx.shape}")

#     # Reconstruct => A * B
#     A = U_approx @ torch.diag(S_approx)  # => [1536, rank]
#     B = Vh_approx                        # => [rank, 768]

#     print(f"[DEBUG] A shape: {A.shape}, B shape: {B.shape}")

#     # Copy => wkv_b.weight, wkv_a.weight
#     with torch.no_grad():
#         print(f"[DEBUG] mla_attn.wkv_b.weight shape: {mla_attn.wkv_b.weight.shape}")
#         print(f"[DEBUG] mla_attn.wkv_a.weight shape: {mla_attn.wkv_a.weight.shape}")

#         assert mla_attn.wkv_b.weight.shape == (A.shape[0], A.shape[1]), (
#             f"Expected wkv_b.weight to be {A.shape}, got {mla_attn.wkv_b.weight.shape}."
#         )
#         mla_attn.wkv_b.weight.copy_(A)

#         assert mla_attn.wkv_a.weight.shape == (B.shape[0], B.shape[1]), (
#             f"Expected wkv_a.weight to be {B.shape}, got {mla_attn.wkv_a.weight.shape}."
#         )
#         mla_attn.wkv_a.weight.copy_(B)

#     # Optionally set kv_norm to identity
#     if hasattr(mla_attn, "kv_norm") and mla_attn.kv_norm is not None:
#         if mla_attn.kv_norm.weight.shape[0] == rank:
#             print("[DEBUG] Setting kv_norm.weight to 1.0 (identity).")
#             with torch.no_grad():
#                 mla_attn.kv_norm.weight.fill_(1.0)

#     # -------------------------------------------------
#     # (C) Copy c_proj => MLA.wo
#     # -------------------------------------------------
#     c_proj_weight = gpt2_sdpa_attn.c_proj.weight  # [768, 768]
#     c_proj_bias   = gpt2_sdpa_attn.c_proj.bias    # [768]

#     print(f"[DEBUG] c_proj_weight shape: {c_proj_weight.shape}")
#     if c_proj_bias is not None:
#         print(f"[DEBUG] c_proj_bias shape: {c_proj_bias.shape}")

#     with torch.no_grad():
#         print(f"[DEBUG] mla_attn.wo.weight shape: {mla_attn.wo.weight.shape}")
#         assert mla_attn.wo.weight.shape == (embed_dim, embed_dim), (
#             f"Expected MLA wo.weight to be [{embed_dim}, {embed_dim}] but got {mla_attn.wo.weight.shape}."
#         )
#         mla_attn.wo.weight.copy_(c_proj_weight.T)

#         if mla_attn.wo.bias is not None and c_proj_bias is not None:
#             print("[DEBUG] Copying c_proj bias...")
#             print(f"[DEBUG] mla_attn.wo.bias shape: {mla_attn.wo.bias.shape}")
#             mla_attn.wo.bias.copy_(c_proj_bias)

#     print("[DEBUG] transform_gpt2sdpa_to_mla completed successfully.")

#     return mla_attn

# def transform_gpt2sdpa_to_mla(
#     gpt2_block: GPT2Block,
#     mla_attn: "MLA",  # your MLA class instance
# ):
#     """
#     Transforms (copies/factorizes) weights from a GPT2SdpaAttention
#     into the given MLA instance, ensuring dtype consistency and handling SVD dtype issues.
#     """
#     gpt2_sdpa_attn: GPT2SdpaAttention = gpt2_block.attn
#     embed_dim = gpt2_sdpa_attn.embed_dim

#     # Get c_attn weights and biases, ensuring correct dtype
#     target_dtype = mla_attn.wq.weight.dtype  # Ensure everything is in the same dtype
#     c_attn_weight = gpt2_sdpa_attn.c_attn.weight.to(target_dtype)
#     c_attn_bias = (
#         gpt2_sdpa_attn.c_attn.bias.to(target_dtype)
#         if gpt2_sdpa_attn.c_attn.bias is not None
#         else None
#     )

#     # Split into Q, K, V
#     q_weight, k_weight, v_weight = torch.split(c_attn_weight, embed_dim, dim=1)
#     if c_attn_bias is not None:
#         q_bias, k_bias, v_bias = torch.split(c_attn_bias, embed_dim, dim=0)
#     else:
#         q_bias = k_bias = v_bias = None

#     # Copy Q weights and biases
#     with torch.no_grad():
#         mla_attn.wq.weight.copy_(q_weight.T)
#         if q_bias is not None:
#             if mla_attn.wq.bias is not None:
#                 mla_attn.wq.bias.copy_(q_bias)
#             else:
#                 mla_attn.wq.bias = torch.nn.Parameter(q_bias.clone())

#     # Factorize K + V => SVD
#     kv_weight = torch.cat([k_weight, v_weight], dim=0).to(torch.float32)  # Convert to float32 for SVD
#     rank = mla_attn.kv_lora_rank

#     # Perform SVD in float32
#     U, S, Vh = torch.linalg.svd(kv_weight, full_matrices=False)
#     U_approx = U[:, :rank]
#     S_approx = S[:rank]
#     Vh_approx = Vh[:rank, :]

#     A = U_approx @ torch.diag(S_approx)
#     B = Vh_approx

#     # Convert back to original dtype if needed
#     A = A.to(target_dtype)
#     B = B.to(target_dtype)

#     with torch.no_grad():
#         mla_attn.wkv_b.weight.copy_(A)
#         mla_attn.wkv_a.weight.copy_(B)

#     # Optionally set kv_norm to identity
#     if hasattr(mla_attn, "kv_norm") and mla_attn.kv_norm is not None:
#         with torch.no_grad():
#             mla_attn.kv_norm.weight.fill_(1.0)
#             if hasattr(mla_attn.kv_norm, "bias") and mla_attn.kv_norm.bias is not None:
#                 mla_attn.kv_norm.bias.fill_(0.0)

#     # Copy c_proj weights and biases
#     c_proj_weight = gpt2_sdpa_attn.c_proj.weight.to(target_dtype)
#     c_proj_bias = (
#         gpt2_sdpa_attn.c_proj.bias.to(target_dtype)
#         if gpt2_sdpa_attn.c_proj.bias is not None
#         else None
#     )

#     with torch.no_grad():
#         mla_attn.wo.weight.copy_(c_proj_weight.T)
#         if c_proj_bias is not None:
#             if mla_attn.wo.bias is not None:
#                 mla_attn.wo.bias.copy_(c_proj_bias)
#             else:
#                 mla_attn.wo.bias = torch.nn.Parameter(c_proj_bias.clone())

#     return mla_attn


def transform_gpt2sdpa_to_mla(
    gpt2_block: GPT2Block,
    mla_attn: "MLA",
):
    gpt2_sdpa_attn: GPT2SdpaAttention = gpt2_block.attn
    embed_dim = gpt2_sdpa_attn.embed_dim

    target_dtype = mla_attn.wq.weight.dtype
    c_attn_weight = gpt2_sdpa_attn.c_attn.weight.to(target_dtype)
    c_attn_bias = (
        gpt2_sdpa_attn.c_attn.bias.to(target_dtype)
        if gpt2_sdpa_attn.c_attn.bias is not None
        else None
    )

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
    explained_variance = (S**2) / (S**2).sum()
    cumulative_variance = torch.cumsum(explained_variance, dim=0)
    rank = (cumulative_variance < 0.99).sum().item()
    rank = max(rank, 1)

    U_approx = U[:, :rank]
    S_approx = S[:rank]
    Vh_approx = Vh[:rank, :]

    A = U_approx @ torch.diag(S_approx)
    B = Vh_approx

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

    A = A / torch.norm(A, dim=-1, keepdim=True)
    B = B / torch.norm(B, dim=-1, keepdim=True)

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
    c_proj_bias = (
        gpt2_sdpa_attn.c_proj.bias.to(target_dtype)
        if gpt2_sdpa_attn.c_proj.bias is not None
        else None
    )

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
    gpt2sdpa = gpt2_block.attn  # GPT2SdpaAttention
    layernorm2 = gpt2_block.ln_2
    gpt2_mlp = gpt2_block.mlp  # GPT2MLP

    # 1) Retrieve the MGQA attention module
    mgqa_norms, mgqa_block, mgqa_residual = mgqa.layers[0]
    mgqa_attn = mgqa_block

    # 2) GPT2SdpaAttention: gather shapes & param references
    embed_dim = gpt2sdpa.embed_dim
    c_attn_weight = gpt2sdpa.c_attn.weight  # shape [embed_dim, 3*embed_dim]
    c_attn_bias = gpt2sdpa.c_attn.bias  # shape [3*embed_dim]
    c_proj_weight = gpt2sdpa.c_proj.weight  # shape [embed_dim, embed_dim]
    c_proj_bias = gpt2sdpa.c_proj.bias  # shape [embed_dim]

    # 3) Split out the Q/K/V weights from c_attn

    with torch.no_grad():
        q_w, k_w, v_w = torch.split(c_attn_weight, embed_dim, dim=1)
        mgqa_attn.to_q.weight.copy_(
            q_w
        )  # query size remain identical, no change needed

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

    mgqa_attn.attend.attn_dropout.p = (
        gpt2sdpa.attn_dropout.p
    )  # Copy over dropout probability
    mgqa_attn.attend.causal = True  # GPT-2 standard

    return mgqa


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
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None]:

        # (1) Convert inputs to the dtype MLA uses
        if hidden_states is not None:
            hidden_states = hidden_states.to(torch.bfloat16)
            bsz, seqlen, _ = hidden_states.shape
        else:
            bsz, seqlen = 0, 0
        desired_seqlen = 512

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
            if (
                hasattr(self.mla, attr_name)
                and getattr(self.mla, attr_name) is not None
            ):
                setattr(
                    self.mla, attr_name, getattr(self.mla, attr_name).to(torch.bfloat16)
                )

        # (4) Always create a valid freqs_cis, even if rope_dim=0
        rope_dim = getattr(self.mla, "qk_rope_head_dim", 0)
        if rope_dim > 0:
            # Normal case if rope_dim>0
            dummy_freq = torch.zeros(
                (seqlen, rope_dim // 2),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            dummy_ones = torch.ones_like(dummy_freq, dtype=torch.float32)
            freqs_cis_fp32 = torch.polar(
                dummy_ones, dummy_freq
            )  # shape: [seqlen, rope_dim//2], complex float32
            freqs_cis = freqs_cis_fp32.to(torch.bfloat16)
        else:
            # Rope dim is 0, but MLA code still calls apply_rotary_emb.
            # Pass an EMPTY complex tensor instead of None to avoid the crash.
            # shape => [seqlen, 0] so .view() won't fail
            freqs_cis = torch.zeros(
                (seqlen, 0), dtype=torch.complex32, device=hidden_states.device
            )

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

        return (output, None, None)


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


init_func_map = {"mla": gpt2sdpa_to_mla_init, "mgqa": gpt2sdpa_to_mgqa_init}
transform_func_map = {
    "mla": transform_gpt2sdpa_to_mla,
    "mgqa": transform_gpt2sdpa_to_mgqa,
}
wrapper_map = {
    "mla": MLAWrapper,
    "mgqa": MGQAWrapper,
}
