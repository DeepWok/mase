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

def spda_transform_pass(model):
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


def test_single_attn(model):
    model = model.to(device)
    model.eval()
    sdpa_attn = extract_gpt2sdpa(model, layer_index=0)  # pick a layer
    sdpa_attn.eval()
    for p in sdpa_attn.parameters():
        torch.nn.init.normal_(p, mean=0, std=0.02)
    
    batch_size, seq_len, embed_dim = 2, 5, sdpa_attn.embed_dim
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    hidden_states = hidden_states.to(device)

    with torch.no_grad():
        orig_out, _, _ = sdpa_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            use_cache=False,
            output_attentions=False
        )

    mgqa_cfg = {"causal": True}
    mgqa_module = gpt2sdpa_to_mgqa_init(sdpa_attn, mgqa_cfg)
    mgqa_module.eval()
    mgqa_module = transform_gpt2sdpa_to_mgqa(sdpa_attn, mgqa_module)
    mgqa_module = mgqa_module.to(device)

    with torch.no_grad():
        mgqa_out = mgqa_module(x=hidden_states)
    
    diff = (orig_out - mgqa_out).abs().max().item()
    print("GPT2Sdpa output shape:", orig_out.shape)
    print("MGQA output shape:", mgqa_out.shape)
    print(f"Max difference: {diff:.6f}")
    assert diff < 1e-4, "Outputs differ too much!"

def test_whole_model(model):
    model = model.to(device)
    model.eval()
    sample_text = "Hello, how are you today?"
    tokenized = tokenizer.encode(sample_text, return_tensors="pt")  # shape: [batch_size=1, sequence_length]
    tokenized = tokenized.to(device)

    # Pass those token IDs into the model:
    with torch.no_grad():
        output = model(tokenized)
    gpt2logits = output.logits

    model = spda_transform_pass(model).to(device)
    with torch.no_grad():
        output = model(tokenized)
    mgqalogits = output.logits

def test_attn_from_model(model):
    model = model.to(device)
    model.eval()
    sdpa_attn = extract_gpt2sdpa(model, layer_index=0)  # pick a layer
    sdpa_attn.eval()

    model = spda_transform_pass(model).to(device)
    mgqa_module = extract_gpt2sdpa(model, layer_index=0).mgqa
    mgqa_module.eval()
    mgqa_module = mgqa_module.to(device)

    # test by performing inference
    batch_size, seq_len, embed_dim = 2, 5, sdpa_attn.embed_dim
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)
    hidden_states = hidden_states.to(device)

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

    # further tesing
    embed_dim = sdpa_attn.embed_dim
    c_attn_weight = sdpa_attn.c_attn.weight
    c_attn_bias   = sdpa_attn.c_attn.bias  

    q_w, k_w, v_w = torch.split(c_attn_weight, embed_dim, dim=1)
    q_w, k_w, v_w = q_w.transpose(0, 1), k_w.transpose(0, 1), v_w.transpose(0, 1)
    q_b, k_b, v_b = torch.split(c_attn_bias, embed_dim, dim=0)
    # x, q_out, k_out, v_out = sdpa_attn.x, sdpa_attn.q, sdpa_attn.k, sdpa_attn.v

    m_qw, m_qb = mgqa_module.to_q.weight, mgqa_module.to_q.bias
    m_kw, m_kb = mgqa_module.to_k.weight, mgqa_module.to_k.bias
    m_vw, m_vb = mgqa_module.to_v.weight, mgqa_module.to_v.bias

    # mx, mq_out, nk_out, nv_out = mgqa_module.x, mgqa_module.q, mgqa_module.k, mgqa_module.v

    print("q_w:", torch.equal(q_w, m_qw))
    print("k_w:", torch.equal(k_w, m_kw))
    print("v_w:", torch.equal(v_w, m_vw))
    # print("q_out:", torch.allclose(q_out, mq_out))
    # print("k_out:", torch.allclose(k_out, nk_out))
    # print("v_out:", torch.allclose(v_out, nv_out))
    # print("x:", torch.allclose(x, mx))
    print("q_b:", torch.equal(q_b, m_qb))
    print("k_b:", torch.equal(k_b, m_kb))
    print("v_b:", torch.equal(v_b, m_vb))




def get_intermediates(model):
    model = model.to(device)
    handles = []
    intermediate_outputs = []

    def make_hook(name):
        """
        Return a function that can be used as a forward hook for a module.
        It will store the output in a dict, keyed by the module's name.
        """
        def hook(module, input, output):
            intermediate_outputs.append((name, output))
        return hook

    # Register hooks on submodules
    suffixes = (".attn", ".mlp", "drop", "ln_1")
    for name, module in model.named_modules():
        if any(name.endswith(suf) for suf in suffixes):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)
    
    # feed sample input in model
    sample_text = "Hello, how are you today?"
    tokenized = tokenizer.encode(sample_text, return_tensors="pt")  # shape: [batch_size=1, sequence_length]
    tokenized = tokenized.to(device)

    # Pass those token IDs into the model:
    with torch.no_grad():
        output = model(tokenized)
    gpt2logits = output.logits

    # Clean up hooks
    for h in handles:
        h.remove()
    
    return intermediate_outputs

def analysis(data1, data2):
    """
    Compare layer-by-layer outputs of two models, storing or printing metrics.
    :param data1: List of (layer_name, output_tensor) from model 1
    :param data2: List of (layer_name, output_tensor) from model 2
    """
    # Sanity check that both lists have the same length
    if len(data1) != len(data2):
        print(f"Warning: data1 has {len(data1)} items, data2 has {len(data2)} items.")
    
    num_pairs = min(len(data1), len(data2))
    
    for i in range(num_pairs):
        name1, out1 = data1[i]
        name2, out2 = data2[i]
        if isinstance(out1, tuple):
            out1 = out1[0]
        if isinstance(out2, tuple):
            out2 = out2[0]

        # Compare shape
        shape1 = out1.shape if isinstance(out1, torch.Tensor) else None
        shape2 = out2.shape if isinstance(out2, torch.Tensor) else None
        
        print(f"\n--- Layer {i} ---")
        print(f"Model1 layer name: {name1}, shape: {shape1}")
        print(f"Model2 layer name: {name2}, shape: {shape2}")
        
        if shape1 != shape2:
            print("  Shape mismatch!")
            continue  # No point in numeric comparison if shapes differ

        # Compute numeric differences
        

        diff = out1 - out2
        max_abs_diff = diff.abs().max().item()
        mean_abs_diff = diff.abs().mean().item()
        
        # A norm-based measure:
        l2_diff = diff.norm(2).item()
        print(f"  max |delta|: {max_abs_diff:.6g}")
        print(f"  mean|delta|: {mean_abs_diff:.6g}")
        print(f"  L2 norm of delta: {l2_diff:.6g}")



if __name__ == "__main__":
    with open(f"{Path.home()}/Projects/mase/mase_output/bert-uncased-2epoch.pkl", "rb") as f:
        model = dill.load(f)
    # test_single_attn(model)
    test_attn_from_model(model)

    # gpt2_inter = get_intermediates(model)
    # model = spda_transform_pass(model).to(device)
    # mgqa_inter = get_intermediates(model)
    # analysis(gpt2_inter, mgqa_inter)



    # test_single_attn()

    

