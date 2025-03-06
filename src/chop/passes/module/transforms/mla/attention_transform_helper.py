import torch
from typing import Optional
import math
from typing import Optional, Tuple, Union

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

def bert_sdpa_to_mla_init(
    bert_attn: BertAttention,  # a BertSdpaSelfAttention object
    config: dict                       # e.g. {"config": some_config}, or any custom dictionary
) -> MLA:
    """
    Initialize and return an MLA module based on dimensions extracted
    from the BertSdpaSelfAttention and/or a custom config.
    This does NOT copy weights from the BERT attention layer; it merely
    sets up an MLA of matching size. All MLA parameters will be newly
    (randomly) initialized.
    """
    sdpa_attn = bert_attn.self

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

def gpt2sdpa_to_mgqa_init(
    gpt2_sdpa_attn: GPT2SdpaAttention,  
    config: dict                       
) -> MGQALayers:
    # Basic info from gpt2_sdpa_attn
    hidden_size = gpt2_sdpa_attn.embed_dim
    num_heads = gpt2_sdpa_attn.num_heads
    attn_drop = gpt2_sdpa_attn.attn_dropout.p
    
    # Set up parameters for MGQALayers, allowing config to overwrite defaults
    mgqa_kwargs = {
        "dim": hidden_size,
        "heads": num_heads,
        "causal": True,
        "depth": config.get("depth", 1),
        "dropout": config.get("dropout", attn_drop),
        "flash": config.get("flash", False),
        "talking_heads": config.get("talking_heads", False),
        "head_scale": config.get("head_scale", False),
        "qk_norm": config.get("qk_norm", False),
        "zero_init_output": config.get("zero_init_output", False),
        # Add any other MGQALayers parameters as needed...
    }
    return MGQALayers(**mgqa_kwargs)

def transform_bert_sdpa_to_mla(
    bert_attn: BertAttention,
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
    sdpa_attn = bert_attn.self
    bert_output = bert_attn.output
    # ------------------------------------------------------------------
    # 0. Sanity checks (world_size assumed = 1, matching dims, etc.)
    # ------------------------------------------------------------------
    bert_hidden_size = sdpa_attn.query.in_features  # Typically 768
    assert mla_attn.dim == bert_hidden_size, (
        f"Inconsistent 'dim' in MLA ({mla_attn.dim}) vs. BERT hidden_size ({bert_hidden_size})."
    )

    # 1. Copy the Q projection directly if q_lora_rank = 0
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

def transform_gpt2sdpa_to_mgqa(
    gpt2sdpa: GPT2SdpaAttention,
    mgqa: MGQALayers,
):
    """
    Load weights and related info from a single GPT2SdpaAttention into the first MGQA attention layer.
    Assumes MGQALayers was set up with depth=1 or has at least one 'a' (attention) block.
    """
    
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
    # MGQA internally uses:
    #   self.to_q = nn.Linear(dim, q_dim, bias=False)
    #   self.to_k = nn.Linear(dim, k_dim, bias=False)
    #   self.to_v = nn.Linear(dim, v_dim, bias=False)
    # assume q_dim = k_dim = v_dim = embed_dim
    # We assume mgqa_attn has bias=False on these. We'll ignore c_attn bias or set them to 0 if needed.

    with torch.no_grad():
        # Slicing out Q/K/V, each shape [d, d]
        q_weight, k_weight, v_weight = torch.split(c_attn_weight, embed_dim, dim=1)

        # Copy to MGQA (MGQA expects (out_features, in_features) as standard for nn.Linear)
        mgqa_attn.to_q.weight.copy_(q_weight)
        mgqa_attn.to_k.weight.copy_(k_weight)
        if mgqa_attn.to_v is not None:
            mgqa_attn.to_v.weight.copy_(v_weight)

        # If your MGQA block has biases on Q/K/V, you could copy them here.
        # By default, from the MGQA code shown, bias=False, so we skip copying.

        # 4) Map c_proj -> mgqa_attn.to_out
        # mgqa_attn.to_out is nn.Linear(out_dim, dim, bias=False) => shape [out_dim, dim]
        # Usually out_dim == embed_dim, dim == embed_dim => shape [embed_dim, embed_dim]
        mgqa_attn.to_out.weight.copy_(c_proj_weight)

        # If your MGQA block uses bias in to_out, you could copy it here as well:
        # if mgqa_attn.to_out.bias is not None:
        #     mgqa_attn.to_out.bias.copy_(c_proj_bias)

    # 5) Copy over dropout probability for the attention module
    # mgqa_attn.attend has 'self.dropout' as the standard dropout param
    mgqa_attn.attend.attn_dropout.p = gpt2sdpa.attn_dropout.p

    # 6) Mark the MGQA attention as causal if needed
    mgqa_attn.attend.causal = True  # GPT-2 standard

    # 7) Possibly copy any other relevant attributes (e.g., scale_attn_weights).
    # That depends on your usage or if you want to keep them in sync.

    # Done. mgqa now has its first layer's attention weights loaded from gpt2sdpa.
    return mgqa

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
            attention_mask = attention_mask.squeeze(1)

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
        return (output, )
    
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
    

init_func_map = {
    "mla": bert_sdpa_to_mla_init,
    "mgqa": gpt2sdpa_to_mgqa_init
}
transform_func_map = {
    "mla": transform_bert_sdpa_to_mla,
    "mgqa": transform_gpt2sdpa_to_mgqa,
}
wrapper_map = {
    "mla": MLAWrapper,
    "mgqa": MGQAWrapper,
}