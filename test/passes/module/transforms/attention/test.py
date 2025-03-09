import torch
import math
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2SdpaAttention
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from copy import deepcopy
from chop.tools import get_tokenized_dataset, get_trainer
from chop import MaseGraph
from pathlib import Path
import chop.passes as passes
import dill
from chop.passes.module.transforms.attention.attention_transform_helper import (
    gpt2sdpa_to_mgqa_init,
    transform_gpt2sdpa_to_mgqa,
)
from chop.passes.module.transforms import quantize_module_transform_pass, attention_transform_pass


device = torch.device("cuda:0")
checkpoint = "openai-community/gpt2"
tokenizer_checkpoint = "openai-community/gpt2"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


def extract_gpt2sdpa(model, layer_index=0):
    return model.transformer.h[layer_index].attn

def test_mla_transform_pass(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "mgqa",
                # "kv_heads": 2,
            }
        },
    }
    model, _ = attention_transform_pass(model, pass_args)
    return model


def test_single_attn():
    with open(f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl", "rb") as f:
        model = dill.load(f)
    model = model.to(device)
    model.eval()
    sdpa_attn = extract_gpt2sdpa(model, layer_index=0)  # pick a layer
    sdpa_attn.eval()
    for p in sdpa_attn.parameters():
        torch.nn.init.normal_(p, mean=0, std=0.02)
    
    batch_size, seq_len, embed_dim = 2, 5, sdpa_attn.embed_dim
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    hidden_states = hidden_states.to(device)

    mgqa_cfg = {"causal": True}
    mgqa_module = gpt2sdpa_to_mgqa_init(sdpa_attn, mgqa_cfg)
    mgqa_module.eval()
    mgqa_module = transform_gpt2sdpa_to_mgqa(sdpa_attn, mgqa_module)
    mgqa_module = mgqa_module.to(device)

    with torch.no_grad():
        orig_out, _, _ = sdpa_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            use_cache=False,
            output_attentions=False
        )
        mgqa_out = mgqa_module(x=hidden_states)
    
    diff = (orig_out - mgqa_out).abs().max().item()
    print("GPT2Sdpa output shape:", orig_out.shape)
    print("MGQA output shape:", mgqa_out.shape)
    print(f"Max difference: {diff:.6f}")
    assert diff < 1e-4, "Outputs differ too much!"

if __name__ == "__main__":
    test_single_attn()

    # with open(f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl", "rb") as f:
    #     model = dill.load(f)
    # model = model.to(device)
    # model.eval()

    # batch_size = 2
    # seq_len = 8
    # input_ids = torch.randint(low=0, high=1000, size=(batch_size, seq_len), device=device)
    # attention_mask = torch.ones_like(input_ids, device=device)

    # with torch.no_grad():
    #     orig_out = model(input_ids=input_ids, attention_mask=attention_mask)

    #     if isinstance(orig_out, tuple):
    #         orig_out_tensor = orig_out[0]
    #     elif hasattr(orig_out, "last_hidden_state"):
    #         orig_out_tensor = orig_out.last_hidden_state
    #     else:
    #         orig_out_tensor = orig_out
    
    # model_transformed = test_mla_transform_pass(model)
    # model_transformed.to(device)
    # model_transformed.eval()

    # with torch.no_grad():
    #     new_out = model_transformed(input_ids=input_ids, attention_mask=attention_mask)

    #     if isinstance(new_out, tuple):
    #         new_out_tensor = new_out[0]
    #     elif hasattr(new_out, "last_hidden_state"):
    #         new_out_tensor = new_out.last_hidden_state
    #     else:
    #         new_out_tensor = new_out

    # # 5) Compare results
    # diff = (orig_out_tensor - new_out_tensor).abs().max().item()
    # print("Original output shape:", orig_out_tensor.shape)
    # print("Transformed output shape:", new_out_tensor.shape)
    # print(f"Max difference: {diff:.6f}")
    # assert diff < 1e-4, "Outputs differ too much after transform pass!"
    



# def test_gpt2sdpa_to_mgqa_correctness():
#     embed_dim = 256
#     n_heads   = 4

#     gpt2_config = GPT2Config(
#         n_embd=embed_dim,
#         n_head=n_heads,
#     )

#     gpt2_sdpa = GPT2SdpaAttention(
#         config=gpt2_config,
#         layer_idx=0,  # just some dummy
#     )
#     gpt2_sdpa.eval()  # no dropout
#     for param in gpt2_sdpa.parameters():
#         torch.nn.init.normal_(param, mean=0, std=0.02)

#     # Make a random input
#     batch_size = 2
#     seq_len    = 5
#     hidden_states = torch.randn(batch_size, seq_len, embed_dim)

#     # Create an MGQA module
#     mgqa_cfg = {
#         "causal": True,
#     }
#     mgqa = gpt2sdpa_to_mgqa_init(gpt2_sdpa, mgqa_cfg)
#     mgqa.eval()  # turn off dropout
#     mgqa = transform_gpt2sdpa_to_mgqa(gpt2_sdpa, mgqa)
    
#     # evaluate
#     with torch.no_grad():
#         orig_out, _, _ = gpt2_sdpa(
#             hidden_states=hidden_states,
#             attention_mask=None,
#             use_cache=False,
#             output_attentions=False
#         )
#     with torch.no_grad():
#         mgqa_out = mgqa(x=hidden_states)  # no cross-attn => context=None

#     # Compare outputs
#     print("Original GPT2Sdpa output shape:", orig_out.shape)
#     print("MGQA output shape:", mgqa_out.shape)

#     # measure the max difference
#     max_diff = (orig_out - mgqa_out).abs().max().item()
#     print(f"Max difference between GPT2Sdpa and MGQA outputs: {max_diff:.6f}")
#     assert max_diff < 1e-4, "Outputs differ more than expected!"
