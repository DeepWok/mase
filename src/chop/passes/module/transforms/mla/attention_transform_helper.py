import torch
from typing import Optional
import math

from chop.nn.mla.modules.model import (
    ModelArgs, 
    MLA,
    RMSNorm
)
from ...module_modify_helper import (
    get_module_by_name, 
    set_module_by_name,
)
from transformers.models.bert.modeling_bert import (
    BertSelfAttention, 
    BertSdpaSelfAttention,
    BertSelfOutput,
)


class MLAWrapper(torch.nn.Module):
    def __init__(self, mla):
        super().__init__()
        self.mla = mla
        # Track or pre-compute any needed start_pos or freqs_cis here if required

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs
    ):
        start_pos = 0
        if hidden_states is not None:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
            bsz, seqlen, _ = hidden_states.shape
        else:
            seqlen = 0  # or handle the None case as you see fit
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bfloat16)
        
        rope_dim = getattr(self.mla, "qk_rope_head_dim", 64)
        dummy_freq = torch.zeros(
            seqlen, rope_dim // 2,
            dtype=torch.float32,       # must be a supported type
            device=hidden_states.device
        )
        dummy_ones = torch.ones_like(dummy_freq, dtype=torch.float32)
        freqs_cis = torch.polar(dummy_ones, dummy_freq)  # This is now float32
        # freqs_cis = freqs_cis.to(dtype=torch.bfloat16)
        
        # noted the output is bf16 format
        output = self.mla(
            x=hidden_states,
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            mask=attention_mask,
        )

        # Convert MLA output back to float32 for subsequent layers
        output = output.to(dtype=torch.float32)
        return output

def instantiate_attention_module(module, postfix, module_map, additional_module_args):
    sdpa_attn = module.self
    self_output = module.output

    additional_module_args = additional_module_args["config"]
    attention_cls = module_map[f"attention_{postfix}"]
    
    attention_module = bert_sdpa_to_mla_init(
        sdpa_attn,
        config=additional_module_args,
    )

    return attention_module


def bert_sdpa_to_mla_init(
    sdpa_attn: BertSdpaSelfAttention,  # a BertSdpaSelfAttention object
    config: dict                       # e.g. {"config": some_config}, or any custom dictionary
) -> MLA:
    """
    Initialize and return an MLA module based on dimensions extracted
    from the BertSdpaSelfAttention and/or a custom config.
    This does NOT copy weights from the BERT attention layer; it merely
    sets up an MLA of matching size. All MLA parameters will be newly
    (randomly) initialized.
    """

    hidden_size = sdpa_attn.query.in_features # idden_size
    n_heads = sdpa_attn.num_attention_heads # number_of_heads

    # read from the user’s config.
    user_config = config.get("config", {})

    # 3. Create a ModelArgs object that MLA expects. 
    model_args = ModelArgs(
        dim=hidden_size,               # match BERT hidden size
        n_heads=n_heads,               # match BERT number of heads
        max_batch_size=user_config.get("max_batch_size", 8),
        max_seq_len=user_config.get("max_seq_len", 4096),
        q_lora_rank=user_config.get("q_lora_rank", 0),
        kv_lora_rank=user_config.get("kv_lora_rank", 512),
        qk_nope_head_dim=user_config.get("qk_nope_head_dim", 64),
        qk_rope_head_dim=user_config.get("qk_rope_head_dim", 0),
        v_head_dim=user_config.get("v_head_dim", 64),
        # You can override or pass in any other fields from user_config ...
        # e.g., rope_factor, rope_theta, etc.
    )

    # 4. Now create an MLA module with those arguments. 
    #    By default, it will have randomly initialized parameters.
    mla_module = MLA(model_args)

    # 5. Return the newly constructed MLA
    return mla_module


def replace_attention_by_name(network, name, module):
    
    original = get_module_by_name(network, name)
    sdpa_attn = original.self
    bert_output = original.output

    new = transform_bert_sdpa_to_mla(sdpa_attn, bert_output, module)
    mla_wapper = MLAWrapper(new)
    network = set_module_by_name(network, name, mla_wapper)
    return network


def transform_bert_sdpa_to_mla(
    sdpa_attn: BertSdpaSelfAttention, # a BertSdpaSelfAttention object
    bert_output: BertSelfOutput,
    mla_attn: MLA                     # an MLA object (already initialized)
):
    """
    Approximate transformation of BertSdpaSelfAttention parameters into MLA parameters.

    Assumptions / Hints:
      - MLA should be configured so that:
          n_heads * qk_head_dim == hidden_size
          n_heads * (qk_nope_head_dim + v_head_dim) == 2 * hidden_size
        or an equivalent factorization that represents Q, K, V dimensions.
      - If q_lora_rank=0, then `mla_attn.wq` is a single linear layer matching BERT's query shape.
      - If kv_lora_rank > 0, we do an SVD-based rank-k factorization for the combined K+V.

    Returns:
        mla_attn (MLA): The same MLA object but with its weights overwritten.
    """

    # ------------------------------------------------------------------
    # 0. Sanity checks (world_size assumed = 1, matching dims, etc.)
    # ------------------------------------------------------------------
    bert_hidden_size = sdpa_attn.query.in_features  # Typically 768
    assert mla_attn.dim == bert_hidden_size, (
        f"Inconsistent 'dim' in MLA ({mla_attn.dim}) vs. BERT hidden_size ({bert_hidden_size})."
    )
    
    # For simplicity, we skip a lot of dimension checks here, but you should
    # confirm that n_heads*(qk_nope_head_dim + qk_rope_head_dim) == hidden_size, etc.
    # in your real code.

    # ------------------------------------------------------------------
    # 1. Copy the Q projection directly if q_lora_rank = 0
    # ------------------------------------------------------------------
    # BERT's query is a torch.nn.Linear: shape (out=hidden_size, in=hidden_size)
    # MLA's wq is ColumnParallelLinear: shape (out=n_heads*qk_head_dim, in=dim).
    # If those shapes match 1:1 (and world_size=1), we can do a direct copy.
    if mla_attn.q_lora_rank == 0:
        with torch.no_grad():
            # Copy weights
            mla_attn.wq.weight.copy_(sdpa_attn.query.weight.data)
            # If you enabled bias in MLA, copy the bias (BERT does have a Q bias by default)
            if mla_attn.wq.bias is not None and sdpa_attn.query.bias is not None:
                mla_attn.wq.bias.copy_(sdpa_attn.query.bias.data)
    else:
        # If q_lora_rank > 0, you'd do a similar factorization approach, or
        # split the Q weight into pieces. Not shown here.
        raise NotImplementedError(
            "This example only handles the q_lora_rank=0 case for the Q projection."
        )

    # ------------------------------------------------------------------
    # 2. Factorize K + V together into wkv_a + kv_norm + wkv_b
    # ------------------------------------------------------------------
    # BERT's key and value are also standard nn.Linear layers with shape (768,768).
    # We'll combine them (concatenate along output-dim) -> shape = (768+768=1536, 768).
    # Then we do a rank-R factorization via SVD or any low-rank approximation.
    k_weight = sdpa_attn.key.weight.data
    v_weight = sdpa_attn.value.weight.data
    # shape => (1536, 768)
    kv_weight = torch.cat([k_weight, v_weight], dim=0)

    # Optional: also combine biases or handle them separately. For simplicity:
    k_bias = sdpa_attn.key.bias.data if sdpa_attn.key.bias is not None else None
    v_bias = sdpa_attn.value.bias.data if sdpa_attn.value.bias is not None else None
    # We'll ignore these biases in the factorization. 
    # If you must keep them, consider augmenting kv_weight or do some workaround.

    # SVD factorization
    #    kv_weight ~ U * S * Vh,  and we take rank = mla_attn.kv_lora_rank
    #    Then M_approx = U[:, :r] * S[:r,:r] * Vh[:r, :]
    #    We store one factor in wkv_a.weight, the other in wkv_b.weight.
    rank = mla_attn.kv_lora_rank
    U, S, Vh = torch.linalg.svd(kv_weight, full_matrices=False)
    U_approx = U[:, :rank]                  # shape (1536, rank)
    S_approx = S[:rank]                     # shape (rank,)
    Vh_approx = Vh[:rank, :]                # shape (rank, 768)

    # Factor = (1536, rank) x (rank, 768)
    # We'll call them A and B for clarity
    A = U_approx @ torch.diag(S_approx)     # shape (1536, rank)
    B = Vh_approx                           # shape (rank, 768)

    # MLA expects:
    #   wkv_a: Linear(dim -> kv_lora_rank + qk_rope_head_dim)
    #          so wkv_a.weight has shape ( (kv_lora_rank + qk_rope_head_dim), dim ).
    #          Typically (64, 768) if rank=64 & qk_rope_head_dim=0
    #
    #   wkv_b: ColumnParallelLinear(kv_lora_rank -> n_heads*(qk_nope_head_dim + v_head_dim))
    #          so wkv_b.weight has shape (n_heads*(...), kv_lora_rank),
    #          e.g. (1536, 64) in our example.
    #
    # Because PyTorch’s F.linear uses (out_features, in_features) as the shape of .weight,
    # we assign:
    #   wkv_b.weight = A   with shape = (1536, 64)
    #   wkv_a.weight = B^T with shape = (64, 768)

    with torch.no_grad():
        # Make sure shapes match your MLA config before copying!
        mla_attn.wkv_b.weight.copy_(A)            # shape (1536, 64)
        mla_attn.wkv_a.weight.copy_(B)            # shape (64, 768)
        # If you wish to approximate biases, you could do so here. This example omits it.

    # 2a. Optionally, set kv_norm to identity if you just want minimal interference.
    #     kv_norm is RMSNorm(kv_lora_rank).
    if isinstance(mla_attn.kv_norm, RMSNorm) and mla_attn.kv_norm.weight.shape[0] == rank:
        with torch.no_grad():
            mla_attn.kv_norm.weight.fill_(1.0)  # Essentially no scaling in RMSNorm

    # ------------------------------------------------------------------
    # 3. Copy the final output projection (BERT's `dense`) into MLA's `wo`
    # ------------------------------------------------------------------
    # BERT uses self.out_proj or self.dense with shape (768, 768) as well.
    out_proj = bert_output.dense
    with torch.no_grad():
        mla_attn.wo.weight.copy_(out_proj.weight.data)
        if mla_attn.wo.bias is not None and out_proj.bias is not None:
            mla_attn.wo.bias.copy_(out_proj.bias.data)

    return mla_attn
