import torch
import torch.nn as nn
from typing import Optional
import math
from typing import Optional, Tuple, Union
import logging
import copy

from chop.nn.modules.mla import (
    ModelArgs,
    MLA,
)
from chop.nn.modules.mgqa import MGQALayers, MGQA
from chop.nn.modules.lora_linear import (
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
    GPT2Attention,
    GPT2Block,
)
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer


def instantiate_attention_module(
    module, transform_name, module_map, additional_module_args
):
    additional_module_args = additional_module_args["config"]
    init_func = init_func_map[transform_name]

    attention_module = init_func(
        module,
        config=additional_module_args,
    )

    return attention_module


def replace_attention_by_name(network, name, module, transform_name):
    original = get_module_by_name(network, name)
    transform_func = transform_func_map[transform_name]
    wrapper_class = wrapper_map[transform_name]

    new = transform_func(original, module)
    if wrapper_class != None:
        wapper = wrapper_class(new)
    else:
        wapper = new

    network = set_module_by_name(network, name, wapper)
    return network


def gpt2sdpa_to_mla_init(gpt2_block: GPT2Block, config: dict) -> MLA:
    """
    Initialize and return an MLA module based on dimensions
    extracted from a GPT2Attention (within GPT2Block).

    Args:
        gpt2_block (GPT2Block): A GPT-2 block containing GPT2Attention as `.attn`.
        config (dict): A user config dict, which can contain nested "config" entries
                       for MLA's ModelArgs.
                       e.g. {"config": {"max_batch_size": 8, "q_lora_rank": 0, ...}}
    Returns:
        MLA: A newly constructed MLA module with random initialization.
    """

    # GPT2Block -> GPT2Attention
    gpt2_sdpa_attn: GPT2Attention = gpt2_block.attn

    # gather GPT-2 attention hyperparams
    hidden_size = gpt2_sdpa_attn.embed_dim  # e.g., 768
    n_heads = gpt2_sdpa_attn.num_heads  # e.g., 12
    head_dim = hidden_size // n_heads
    rope_dim = head_dim // 2

    # optional user config
    user_config = config.get("config", {})

    # Create ModelArgs for MLA
    model_args = ModelArgs(
        dim=hidden_size,
        n_heads=n_heads,
        qk_nope_head_dim=head_dim,  # Use full head dimension
        qk_rope_head_dim=0,  # Disable rotary embeddings
        v_head_dim=head_dim,
        max_seq_len=512,  # Match tokenizer max_length
        kv_lora_rank=512,  # Increased rank for full sequences
    )

    # Construct MLA with those arguments
    mla_module = MLA(model_args)

    # Return the newly constructed module (randomly initialized)
    return mla_module


def gpt2sdpa_to_mgqa_init(gpt2_sdpa: GPT2Attention, config: dict) -> MGQA:

    # Basic info from gpt2_sdpa
    hidden_size = gpt2_sdpa.embed_dim
    num_heads = gpt2_sdpa.num_heads
    attn_drop = gpt2_sdpa.attn_dropout.p
    kv_heads = config.get("kv_heads", num_heads)
    kv_heads = num_heads // math.ceil(num_heads / kv_heads)

    mgqa_kwargs = {
        "dim": hidden_size,
        "dim_head": hidden_size // num_heads,
        "heads": num_heads,  # number of query heads
        "kv_heads": kv_heads,  # group or unify the KV heads
        "causal": config.get("causal", True),  # GPT-2 is typically causal
        "dropout": config.get("dropout", attn_drop),
    }
    mgqa_layers = MGQA(**mgqa_kwargs)
    return mgqa_layers


def gpt2sdpa_to_lorafc_init(attn_module: GPT2Attention, config: dict) -> nn.Module:
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
    mla_attn: "MLA",
):
    gpt2_sdpa_attn: GPT2Attention = gpt2_block.attn
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
    gpt2sdpa: GPT2Attention,
    mgqa: MGQA,
):

    # GPT2Attention: gather shapes & param references
    embed_dim = gpt2sdpa.embed_dim
    c_attn_weight = gpt2sdpa.c_attn.weight  # shape [embed_dim, 3*embed_dim]
    c_attn_bias = gpt2sdpa.c_attn.bias  # shape [3*embed_dim]
    c_proj_weight = gpt2sdpa.c_proj.weight  # shape [embed_dim, embed_dim]
    c_proj_bias = gpt2sdpa.c_proj.bias  # shape [embed_dim]

    # MGQA: gather shapes & param references
    heads = mgqa.heads
    kv_heads = mgqa.kv_heads

    # 3) Split out the Q/K/V weights from c_attn
    with torch.no_grad():
        q_w, k_w, v_w = torch.split(c_attn_weight, embed_dim, dim=1)
        q_w, k_w, v_w = q_w.transpose(0, 1), k_w.transpose(0, 1), v_w.transpose(0, 1)
        q_b, k_b, v_b = torch.split(c_attn_bias, embed_dim, dim=0)

        mgqa.to_q.weight.copy_(q_w)  # query size remain identical, no change needed
        mgqa.to_q.bias.copy_(q_b)

        dim_head = embed_dim // heads

        # handle kv pair reduction
        if kv_heads < heads:

            g_size = heads // kv_heads

            # -------- Weight --------
            # [heads * dim_head, in_dim] -> [heads, dim_head, in_dim]
            k_w_r = k_w.reshape(
                heads, dim_head, embed_dim
            )  # [heads, dim_head, embed_dim]
            v_w_r = v_w.reshape(heads, dim_head, embed_dim)

            # Group the heads => shape [kv_heads, g_size, dim_head, in_dim]
            k_w_r = k_w_r.reshape(kv_heads, g_size, dim_head, embed_dim).mean(dim=1)
            v_w_r = v_w_r.reshape(kv_heads, g_size, dim_head, embed_dim).mean(
                dim=1
            )  # [kv_heads, dim_head, in_dim]

            # Flatten the row dims => shape [kv_heads * dim_head, in_dim]
            k_w = k_w_r.reshape(kv_heads * dim_head, embed_dim)
            v_w = v_w_r.reshape(kv_heads * dim_head, embed_dim)

            # -------- Biases --------
            # original shape is [heads * dim_head].
            k_b_r = k_b.reshape(heads, dim_head)  # [heads, dim_head]
            v_b_r = v_b.reshape(heads, dim_head)
            k_b_r = k_b_r.reshape(kv_heads, g_size, dim_head).mean(
                dim=1
            )  # [kv_heads, dim_head]
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

    mgqa.attend.attn_dropout.p = (
        gpt2sdpa.attn_dropout.p
    )  # Copy over dropout probability

    return mgqa


def transform_gpt2sdpa_to_lorafc(
    gpt2sdpa: GPT2Attention,
    new_fc: nn.Module,
) -> GPT2Attention:

    use_low_rank = isinstance(new_fc, LowRankLinear)

    if use_low_rank:
        rank = new_fc.rank
        with torch.no_grad():
            weight = gpt2sdpa.c_proj.weight.T
            # Apply SVD
            try:
                U, S, V = torch.svd(weight)

                # slice to rank
                U_r = U[:, :rank]
                S_r = S[:rank]
                V_r = V[:, :rank]

                # compute factors directly
                A_weight = V_r * torch.sqrt(S_r)
                B_weight = U_r * torch.sqrt(S_r.unsqueeze(0))

                new_fc.A.weight.copy_(A_weight.t())
                new_fc.B.weight.copy_(B_weight)

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

                attention_mask = attention_mask[
                    :, None, None, :
                ]  # Shape: (bsz, 1, 1, seqlen)
            # Expand mask to match the number of attention heads
            # attention_mask shape [bsz, 1, seqlen, endpos]
            attention_mask = attention_mask.expand(
                -1, self.mla.n_heads, -1, -1
            )  # Shape: (bsz, n_heads, seqlen, endpos)
            # Permute mask to match the expected format for MLA
            attention_mask = attention_mask[:, :, 0, :]  # Shape: (bsz, n_heads, endpos)
            attention_mask = attention_mask.to(torch.bfloat16)

        # Enforce cache dtype before forward pass
        self._init_caches()

        # Generate rotary embeddings
        rope_dim = getattr(self.mla, "qk_rope_head_dim", 0)
        freqs_cis = self._create_rotary_embeddings(
            seqlen, rope_dim, hidden_states.device
        )

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
            dummy_freq = torch.zeros(
                (seqlen, rope_dim // 2), dtype=torch.float32, device=device
            )
            # Convert to polar form (complex numbers) and cast to BFloat16
            return torch.polar(torch.ones_like(dummy_freq), dummy_freq).to(
                torch.bfloat16
            )
        # Return empty tensor if no rotary embeddings are needed
        return torch.zeros((seqlen, 0), dtype=torch.bfloat16, device=device)


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
            input_mask = torch.ones(
                hidden_states.size()[:-1], device=hidden_states.device, dtype=torch.bool
            )
        else:
            # Handle incoming 4D attention_mask from GPT-2
            if attention_mask.dim() == 4:
                input_mask = attention_mask[:, 0, 0, :].bool()
            elif attention_mask.dim() == 2:
                input_mask = attention_mask.bool()
            else:
                raise ValueError(
                    f"Unexpected attention_mask shape: {attention_mask.shape}"
                )

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
    "gpt2block_to_mla": gpt2sdpa_to_mla_init,
    "gpt2spda_to_mgqa": gpt2sdpa_to_mgqa_init,
    "gpt2spda_to_lora_fc": gpt2sdpa_to_lorafc_init,
}

transform_func_map = {
    "gpt2block_to_mla": transform_gpt2sdpa_to_mla,
    "gpt2spda_to_mgqa": transform_gpt2sdpa_to_mgqa,
    "gpt2spda_to_lora_fc": transform_gpt2sdpa_to_lorafc,
}

wrapper_map = {
    "gpt2block_to_mla": MLAWrapper,
    "gpt2spda_to_mgqa": MGQAWrapper,
    "gpt2spda_to_lora_fc": None,  # do not require a wrapper
}


def transform_llama_to_mla(model, config):
    """Apply the three stages of transformations to the model."""
    # Make a deep copy to modify
    model_copy = copy.deepcopy(model)

    hidden_size = model_copy.config.hidden_size
    n_heads = model_copy.config.num_attention_heads
    kv_heads = model_copy.config.num_key_value_heads
    head_dim = model_copy.config.hidden_size // model_copy.config.num_attention_heads
    latent_dim = kv_heads * head_dim
    kv_groups = model_copy.config.num_attention_heads // kv_heads if kv_heads > 0 else 1

    print(f"\nModel architecture details:")
    print(f"hidden_size: {hidden_size}, n_heads: {n_heads}, kv_heads: {kv_heads}")
    print(f"head_dim: {head_dim}, latent_dim: {latent_dim}, kv_groups: {kv_groups}")

    # Stage 1: Initial Weight Modification (Identity Matrices)
    print("\n--- Applying Initial Weight Modification (Identity) ---")
    modified_layers = 0

    with torch.no_grad():
        for name, module in model_copy.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                target_device = module.weight.data.device
                target_dtype = module.weight.data.dtype
                if "k_up_proj" in name or "v_up_proj" in name:
                    modified_layers += 1
                    if modified_layers == 1:
                        print(f"Modifying layer: {name}")

                    # Constructing identity matrix based on dimensions
                    identity_weight = (
                        torch.stack(
                            [
                                torch.eye(latent_dim).reshape(
                                    kv_heads, head_dim, latent_dim
                                )
                            ]
                            * kv_groups,
                            dim=1,
                        )
                        .reshape(hidden_size, latent_dim)
                        .contiguous()
                        .to(target_device, target_dtype)
                    )

                    if "k_up_proj" in name:
                        # Reshape/transpose specific to k_up_proj
                        identity_weight = (
                            identity_weight.view(hidden_size, kv_heads, head_dim)
                            .transpose(1, 2)
                            .contiguous()
                            .view(hidden_size, latent_dim)
                        )

                    module.weight.data.copy_(identity_weight)

                elif "k_proj" in name:  # Apply reshaping to k_proj weights and bias
                    # Reshape weight
                    reshaped_weight = (
                        module.weight.data.view(kv_heads, head_dim, hidden_size)
                        .transpose(0, 1)
                        .contiguous()
                        .view(latent_dim, hidden_size)
                    )
                    module.weight.data.copy_(reshaped_weight)

                    # Reshape bias if it exists
                    if hasattr(module, "bias") and module.bias is not None:
                        reshaped_bias = (
                            module.bias.data.view(kv_heads, head_dim)
                            .transpose(0, 1)
                            .contiguous()
                            .view(latent_dim)
                        )
                        module.bias.data.copy_(reshaped_bias)

    print(f"Initial modification complete. Modified {modified_layers} layers.")

    # Stage 2: Second Weight Modification (Orthogonalization via SVD)
    print("\n--- Applying Second Weight Modification (Orthogonalization) ---")
    modified_layers = 0

    with torch.no_grad():
        for name, module in model_copy.named_modules():
            # Check if the module is a self-attention layer
            if (
                isinstance(module, torch.nn.Module)
                and "self_attn" in name
                and hasattr(module, "k_up_proj")
            ):
                modified_layers += 1
                if modified_layers == 1:
                    print(f"Orthogonalizing layer: {name}")

                target_device = module.q_proj.weight.device
                target_dtype = module.q_proj.weight.dtype

                # Orthogonalize q_proj and k_up_proj
                k_up_weight = module.k_up_proj.weight.data.clone().reshape(
                    n_heads, head_dim, latent_dim
                )
                q_weight = module.q_proj.weight.data.clone().reshape(
                    n_heads, head_dim, hidden_size
                )

                if module.q_proj.bias is not None:
                    q_bias = module.q_proj.bias.data.clone().reshape(
                        n_heads, head_dim, 1
                    )
                    q_weight = torch.cat(
                        [q_weight, q_bias], dim=-1
                    )  # Append bias as a column

                q_k_up = torch.einsum("hdc,hdD->hcD", k_up_weight, q_weight)

                # SVD - Use torch.linalg.svd for stability
                U, S, Vh = torch.linalg.svd(q_k_up, full_matrices=False)
                V = Vh.mH  # Conjugate transpose for V

                # Keep only top 'head_dim' components
                U = U[:, :, :head_dim]
                S = S[:, :head_dim]
                V = V[:, :, :head_dim]

                S_sqrt = torch.sqrt(S)
                US_sqrt = torch.einsum("hLd, hd->hdL", U, S_sqrt)
                S_sqrtV = torch.einsum(
                    "hd, hdD->hdD", S_sqrt, V.mH
                )  # Note the einsum pattern from your updated code

                # Update weights and bias
                module.k_up_proj.weight.data.copy_(
                    US_sqrt.reshape(n_heads * head_dim, latent_dim).contiguous()
                )

                if module.q_proj.bias is not None:
                    module.q_proj.bias.data.copy_(
                        S_sqrtV[:, :, -1].reshape(-1).contiguous()
                    )
                    S_sqrtV_weights = S_sqrtV[
                        :, :, :-1
                    ]  # Separate weights from bias column
                else:
                    S_sqrtV_weights = S_sqrtV

                module.q_proj.weight.data.copy_(
                    S_sqrtV_weights.reshape(
                        n_heads * head_dim, hidden_size
                    ).contiguous()
                )

                # Orthogonalize o_proj and v_up_proj
                v_up_weight = module.v_up_proj.weight.data.clone().reshape(
                    n_heads, head_dim, latent_dim
                )
                o_weight = module.o_proj.weight.data.clone().reshape(
                    hidden_size, n_heads, head_dim
                )
                v_up_o = torch.einsum("hdc,Dhd->hcD", v_up_weight, o_weight)

                # SVD
                U, S, Vh = torch.linalg.svd(v_up_o, full_matrices=False)
                V = Vh.mH

                # Keep only top 'head_dim' components
                U = U[:, :, :head_dim]
                S = S[:, :head_dim]
                V = V[:, :, :head_dim]

                S_sqrt = torch.sqrt(S)
                US_sqrt = torch.einsum("hLd, hd->hdL", U, S_sqrt)
                S_sqrtV = torch.einsum("hd, hDd->Dhd", S_sqrt, V)

                # Update weights
                module.v_up_proj.weight.data.copy_(
                    US_sqrt.reshape(n_heads * head_dim, latent_dim).contiguous()
                )
                module.o_proj.weight.data.copy_(
                    S_sqrtV.reshape(hidden_size, n_heads * head_dim).contiguous()
                )

    print(f"Orthogonalization complete. Modified {modified_layers} layers.")

    # Stage 3: Third Weight Modification (Absorption)
    print("\n--- Applying Third Weight Modification (Absorption) ---")

    with torch.no_grad():
        layers_to_modify = []
        # First, identify layers that still need absorption
        for name, module in model_copy.named_modules():
            if (
                "self_attn" in name
                and hasattr(module, "k_up_proj")
                and hasattr(module, "v_up_proj")
            ):
                layers_to_modify.append((name, module))

        print(f"Found {len(layers_to_modify)} layers to absorb")

        # Now, modify them. This avoids issues with modifying modules while iterating.
        for idx, (name, module) in enumerate(layers_to_modify):
            if idx == 0:
                print(f"Absorbing layer: {name}")

            target_device = module.q_proj.weight.device
            target_dtype = module.q_proj.weight.dtype

            # Absorb k_up_proj into q_proj
            k_up_weight = module.k_up_proj.weight.data.clone().reshape(
                n_heads, head_dim, latent_dim
            )
            q_weight = module.q_proj.weight.data.clone().reshape(
                n_heads, head_dim, hidden_size
            )
            q_bias_data = None

            if module.q_proj.bias is not None:
                q_bias = module.q_proj.bias.data.clone().reshape(n_heads, head_dim, 1)
                q_weight = torch.cat([q_weight, q_bias], dim=-1)  # Append bias column

            q_k_up = torch.einsum("hdc,hdD->hcD", k_up_weight, q_weight)

            # Create new linear layer for absorbed q_proj
            new_q_proj_out_features = n_heads * latent_dim
            new_q_proj = torch.nn.Linear(
                hidden_size,
                new_q_proj_out_features,
                bias=(module.q_proj.bias is not None),
            )
            new_q_proj = new_q_proj.to(device=target_device, dtype=target_dtype)

            if module.q_proj.bias is not None:
                new_q_proj.bias.data.copy_(q_k_up[:, :, -1].reshape(-1).contiguous())
                q_k_up_weights = q_k_up[:, :, :-1]  # Separate weights
            else:
                q_k_up_weights = q_k_up

            new_q_proj.weight.data.copy_(
                q_k_up_weights.reshape(
                    new_q_proj_out_features, hidden_size
                ).contiguous()
            )

            # Replace module's q_proj and delete k_up_proj
            setattr(module, "q_proj", new_q_proj)
            delattr(module, "k_up_proj")

            # Absorb v_up_proj into o_proj
            v_up_weight = module.v_up_proj.weight.data.clone().reshape(
                n_heads, head_dim, latent_dim
            )
            o_weight = module.o_proj.weight.data.clone().reshape(
                hidden_size, n_heads, head_dim
            )
            v_up_o = torch.einsum("hdc,Dhd->Dhc", v_up_weight, o_weight)

            # Create new linear layer for absorbed o_proj
            new_o_proj_in_features = n_heads * latent_dim
            original_o_proj_bias_exists = (
                hasattr(module.o_proj, "bias") and module.o_proj.bias is not None
            )
            new_o_proj = torch.nn.Linear(
                new_o_proj_in_features, hidden_size, bias=original_o_proj_bias_exists
            )
            new_o_proj = new_o_proj.to(device=target_device, dtype=target_dtype)

            new_o_proj.weight.data.copy_(
                v_up_o.reshape(hidden_size, new_o_proj_in_features).contiguous()
            )
            if original_o_proj_bias_exists:
                new_o_proj.bias.data.copy_(module.o_proj.bias.data)

            # Replace module's o_proj and delete v_up_proj
            setattr(module, "o_proj", new_o_proj)
            delattr(module, "v_up_proj")

            # Set flag
            setattr(module, "absorb", True)

    print(f"Absorption complete. Modified {len(layers_to_modify)} layers.")

    return model_copy
