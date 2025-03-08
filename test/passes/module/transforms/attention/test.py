import torch
import math
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2SdpaAttention
from copy import deepcopy
from chop.passes.module.transforms.attention.attention_transform_helper import (
    gpt2sdpa_to_mgqa_init,
    transform_gpt2sdpa_to_mgqa,
)
# from chop.nn.attention.modules.gpt2 import (
#     GPT2Config,
#     GPT2SdpaAttention,
# )

# ============ your transformation code (imports omitted) ============


def test_gpt2sdpa_to_mgqa_correctness():
    embed_dim = 64
    n_heads   = 1

    gpt2_config = GPT2Config(
        n_embd=embed_dim,
        n_head=n_heads,
    )

    gpt2_sdpa = GPT2SdpaAttention(
        config=gpt2_config,
        layer_idx=0,  # just some dummy
    )
    gpt2_sdpa.eval()  # no dropout
    for param in gpt2_sdpa.parameters():
        torch.nn.init.normal_(param, mean=0, std=0.02)

    # Make a random input
    batch_size = 2
    seq_len    = 5
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)

    # Create an MGQA module
    mgqa_cfg = {
        "causal": False,
    }
    mgqa = gpt2sdpa_to_mgqa_init(gpt2_sdpa, mgqa_cfg)
    mgqa.eval()  # turn off dropout
    mgqa = transform_gpt2sdpa_to_mgqa(gpt2_sdpa, mgqa)
    
    # evaluate
    with torch.no_grad():
        orig_out, _, _ = gpt2_sdpa(
            hidden_states=hidden_states,
            attention_mask=None,
            use_cache=False,
            output_attentions=False
        )
    with torch.no_grad():
        mgqa_out = mgqa(x=hidden_states)  # no cross-attn => context=None

    # Compare outputs
    print("Original GPT2Sdpa output shape:", orig_out.shape)
    print("MGQA output shape:", mgqa_out.shape)

    # measure the max difference
    max_diff = (orig_out - mgqa_out).abs().max().item()
    print(f"Max difference between GPT2Sdpa and MGQA outputs: {max_diff:.6f}")
    assert max_diff < 1e-4, "Outputs differ more than expected!"

if __name__ == "__main__":
    test_gpt2sdpa_to_mgqa_correctness()
