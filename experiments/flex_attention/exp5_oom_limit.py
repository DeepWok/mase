"""
Experiment 5: Peak Memory / OOM Limit

Pushes sequence lengths exponentially until the GPU runs out of memory.
Tests both Inference (forward only) and Training (forward + backward) 
to find the absolute context-length limits of each attention backend.

Model: Medium Llama (hidden=2048, 16 layers, 16 heads, 4 kv_heads)
Dtype: bfloat16, Batch Size: 1 (to maximize sequence length)

Usage:
    python -u experiments/flex_attention/exp5_oom_limit.py
"""

import json
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chop.passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)
import chop.passes.module.transforms.attention.flex_attention_transform as fat

import torch._dynamo
import torch._inductor.config

# 1. Prevent Dynamo from falling back to Python eager mode
torch._dynamo.config.cache_size_limit = 128 


# ============================================================================
# Config
# ============================================================================

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 5504
NUM_LAYERS = 16
NUM_HEADS = 16
NUM_KV_HEADS = 4
VOCAB_SIZE = 32000
BATCH_SIZE = 1  # Forced to 1 to find the absolute maximum sequence length
DEVICE = "cuda"
DTYPE = torch.bfloat16
WINDOW_SIZE = 256

# Exponential scaling to find the OOM cliff quickly
SEQ_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# Mask Generators
# ============================================================================

def create_causal_mask(seq_len, dtype, device):
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), min_val, dtype=dtype, device=device)
    q_idx = torch.arange(seq_len, device=device).view(-1, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, -1)
    mask[0, 0, q_idx >= kv_idx] = 0.0
    return mask.contiguous()


def create_sliding_window_mask(seq_len, window_size, dtype, device):
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), min_val, dtype=dtype, device=device)
    q_idx = torch.arange(seq_len, device=device).view(-1, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, -1)
    causal = q_idx >= kv_idx
    in_window = (q_idx - kv_idx) < window_size
    mask[0, 0, causal & in_window] = 0.0
    return mask.contiguous()


# ============================================================================
# Helpers
# ============================================================================

def make_model(max_seq_len, pass_args=None):
    fat._compiled_flex_attention = None

    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=max_seq_len,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)

    if pass_args is not None:
        model, _ = flex_attention_transform_pass(model, pass_args)

    return model


def test_memory_limit(model, seq_len, attention_mask=None, mode="inference"):
    """Runs a single pass to check for OOM."""
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)
    labels = input_ids.clone() if mode == "training" else None

    fwd_kwargs = {}
    if labels is not None:
        fwd_kwargs["labels"] = labels
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask.expand(BATCH_SIZE, -1, -1, -1)

    if mode == "inference":
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            _ = model(input_ids, **fwd_kwargs)
    elif mode == "training":
        model.train()
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, **fwd_kwargs)
            outputs.loss.backward()
    
    torch.cuda.synchronize()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 5: Peak Memory / OOM Limit")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Dtype: {DTYPE}, Batch Size: {BATCH_SIZE}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    configs = [
        ("SDPA causal",              "sdpa",  None, None),
        ("SDPA sliding_window(256)", "sdpa",  None, "sliding_window"),
        ("Flex causal",              "flex",  {"score_mod": "causal"}, None),
        ("Flex sliding_window(256)", "flex",  {"score_mod": "sliding_window",
                                               "score_mod_kwargs": {"window_size": WINDOW_SIZE}}, None),
    ]

    modes = ["inference", "training"]
    
    results = {
        "experiment": "exp5_oom_limit",
        "config": {
            "hidden_size": HIDDEN_SIZE, "batch_size": BATCH_SIZE,
            "window_size": WINDOW_SIZE, "device": torch.cuda.get_device_name(),
        },
        "limits": {mode: {} for mode in modes}
    }

    for mode in modes:
        print("=" * 70)
        print(f"MODE: {mode.upper()}")
        print("=" * 70)

        for label, attn_type, pass_args, mask_type in configs:
            print(f"\nHunting OOM for: {label}")
            print("-" * 50)
            
            last_successful_seq = None
            max_mem_used = None

            for seq_len in SEQ_LENGTHS:
                torch.cuda.reset_peak_memory_stats()
                print(f"  Testing seq_len={seq_len:6d} ... ", end="", flush=True)

                model = None
                attn_mask = None
                
                try:
                    # Creating the manual mask might OOM the GPU before the model even runs!
                    if mask_type == "causal":
                        attn_mask = create_causal_mask(seq_len, DTYPE, DEVICE)
                    elif mask_type == "sliding_window":
                        attn_mask = create_sliding_window_mask(seq_len, WINDOW_SIZE, DTYPE, DEVICE)

                    model = make_model(seq_len, pass_args if attn_type == "flex" else None)
                    
                    # Run the test
                    test_memory_limit(model, seq_len, attention_mask=attn_mask, mode=mode)
                    
                    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                    print(f"PASS (Peak Mem: {peak_mem:.0f} MB)")
                    
                    last_successful_seq = seq_len
                    max_mem_used = peak_mem

                except torch.cuda.OutOfMemoryError:
                    print("OOM!")
                    break  # Stop scaling this config
                except RuntimeError as e:
                    print(f"Error ({str(e).splitlines()[0]})")
                    break
                finally:
                    # Aggressive cleanup
                    if model is not None: del model
                    if attn_mask is not None: del attn_mask
                    torch.cuda.empty_cache()

            results["limits"][mode][label] = {
                "max_seq_len": last_successful_seq,
                "peak_mem_mb": max_mem_used
            }

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Maximum Sequence Length Before OOM")
    print("=" * 70)
    
    col_w = 30
    header = f"{'Config':<{col_w}} | {'Inference Max':>15} | {'Training Max':>15}"
    print(header)
    print("-" * 66)

    for label, *_ in configs:
        inf_max = results["limits"]["inference"][label].get("max_seq_len")
        trn_max = results["limits"]["training"][label].get("max_seq_len")
        
        inf_str = f"{inf_max // 1024}K" if inf_max else "FAILED"
        trn_str = f"{trn_max // 1024}K" if trn_max else "FAILED"
        
        print(f"{label:<{col_w}} | {inf_str:>15} | {trn_str:>15}")

    out_path = RESULTS_DIR / "exp5_oom_limit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()