"""
Experiment 8: Document Masking (Sequence Packing)

Replicates the official PyTorch FlexAttention benchmark.
Packs multiple 1024-token documents into a single sequence. Proves that
FlexAttention can physically skip computing cross-document attention blocks,
resulting in massive speedups over both custom SDPA masks and standard Causal FA2.

Model: Medium Llama (hidden=2048, 16 layers)
Dtype: bfloat16, Forward + Backward
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
BATCH_SIZE = 2
WARMUP_ITERS = 3
BENCH_ITERS = 10
DEVICE = "cuda"
DTYPE = torch.bfloat16

DOC_LEN = 1024  # Each document is strictly 1024 tokens long
SEQ_LENGTHS = [1024, 4096, 8192, 16384]

RESULTS_DIR = Path(__file__).parent / "results"

# ============================================================================
# SDPA Manual Mask Generator
# ============================================================================

def create_document_mask(seq_len, doc_len, dtype, device):
    """Creates a 4D manual mask blocking cross-document attention."""
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), min_val, dtype=dtype, device=device)
    q_idx = torch.arange(seq_len, device=device).view(-1, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, -1)
    
    causal = q_idx >= kv_idx
    same_doc = (q_idx // doc_len) == (kv_idx // doc_len)
    mask[0, 0, causal & same_doc] = 0.0
    return mask.contiguous()

# ============================================================================
# Helpers
# ============================================================================

def make_model(max_seq_len, pass_args=None):
    fat._compiled_flex_attention = None
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS, max_position_embeddings=max_seq_len,
        vocab_size=VOCAB_SIZE, _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)
    if pass_args is not None:
        model, _ = flex_attention_transform_pass(model, pass_args)
    model.train()
    return model

def benchmark_training_latency(model, seq_len, attention_mask=None):
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)
    labels = input_ids.clone()
    fwd_kwargs = {"labels": labels}
    
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask.expand(BATCH_SIZE, -1, -1, -1)

    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(WARMUP_ITERS):
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, **fwd_kwargs)
            outputs.loss.backward()
            
    torch.cuda.synchronize()
    start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    latencies = []
    
    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(BENCH_ITERS):
            model.zero_grad(set_to_none=True)
            start_evt.record()
            outputs = model(input_ids, **fwd_kwargs)
            outputs.loss.backward()
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))

    mean_ms = sum(latencies) / len(latencies)
    return {"mean_ms": mean_ms, "peak_mem_mb": torch.cuda.max_memory_allocated() / 1024 / 1024}

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 8: Document Masking (Sequence Packing)")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Document Size: {DOC_LEN} tokens")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print()

    configs = [
            ("SDPA Causal (FA2 Baseline)", "sdpa", None, False),
            ("SDPA Document Mask", "sdpa", None, True),
            ("Flex Document Mask", "flex", {
                "score_mod": "document_mask", 
                "score_mod_kwargs": {"doc_len": DOC_LEN},
                "mask_mod": "document_mask",
                "mask_mod_kwargs": {"doc_len": DOC_LEN}
            }, False),
        ]

    results = {"experiment": "exp8_document_masking", "benchmarks": {}}

    for label, attn_type, pass_args, use_manual_mask in configs:
        print(f"\nBenchmarking: {label}")
        print("-" * 50)
        results["benchmarks"][label] = {}

        for seq_len in SEQ_LENGTHS:
            torch.cuda.reset_peak_memory_stats()
            print(f"  seq_len={seq_len:5d} ... ", end="", flush=True)

            model = make_model(seq_len, pass_args if attn_type == "flex" else None)
            attn_mask = create_document_mask(seq_len, DOC_LEN, DTYPE, DEVICE) if use_manual_mask else None
            
            try:
                timing = benchmark_training_latency(model, seq_len, attention_mask=attn_mask)
                print(f"mean={timing['mean_ms']:8.2f}ms  mem={timing['peak_mem_mb']:.0f}MB")
                results["benchmarks"][label][str(seq_len)] = timing
            except torch.cuda.OutOfMemoryError:
                print("OOM")
            except RuntimeError as e:
                print(f"Error ({str(e).splitlines()[0]})")
                
            del model
            if attn_mask is not None: del attn_mask
            torch.cuda.empty_cache()

    # --- Summary ---
    col_w = 30
    print("\n" + "=" * 70)
    print("SUMMARY: Training Latency (mean ms)")
    print("=" * 70)
    header = f"{'Config':<{col_w}}"
    for sl in SEQ_LENGTHS: header += f" {sl:>8}"
    print(header)
    print("-" * (col_w + 9 * len(SEQ_LENGTHS)))

    for label, *_ in configs:
        row = f"{label:<{col_w}}"
        for sl in SEQ_LENGTHS:
            val = results["benchmarks"][label].get(str(sl), {}).get("mean_ms")
            row += f" {val:>8.2f}" if val is not None else f" {'OOM':>8}"
        print(row)

    print("\nSPEEDUP (Flex vs SDPA Document Mask):")
    print("-" * (col_w + 9 * len(SEQ_LENGTHS)))
    row = f"{'Flex Speedup':<{col_w}}"
    for sl in SEQ_LENGTHS:
        sdpa_val = results["benchmarks"]["SDPA Document Mask"].get(str(sl), {}).get("mean_ms")
        flex_val = results["benchmarks"]["Flex Document Mask"].get(str(sl), {}).get("mean_ms")
        row += f" {sdpa_val / flex_val:>7.2f}x" if (sdpa_val and flex_val) else f" {'N/A':>8}"
    print(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp8_document_masking.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()