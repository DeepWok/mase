"""
Experiment 4: Mistral Architecture Integration

Compares Hugging Face's native Mistral SWA (via SDPA) against our 
FlexAttention sliding window implementation. Proves the transform pass 
can successfully hijack and accelerate natively sparse architectures.

Model: Medium Mistral (hidden=2048, 16 layers, 16 heads, 4 kv_heads)
Dtype: bfloat16, forward pass only

Usage:
    python -u experiments/flex_attention/exp4_mistral_swa.py
"""

import json
import sys
from pathlib import Path

import torch
from transformers import MistralConfig, MistralForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chop.passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)
import chop.passes.module.transforms.attention.flex_attention_transform as fat

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

# ============================================================================
# Config
# ============================================================================

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 5632
NUM_LAYERS = 16
NUM_HEADS = 16
NUM_KV_HEADS = 4
VOCAB_SIZE = 32000
BATCH_SIZE = 2
WARMUP_ITERS = 3
BENCH_ITERS = 10
DEVICE = "cuda"
DTYPE = torch.bfloat16
WINDOW_SIZE = 256

SEQ_LENGTHS = [256, 512, 1024, 2048, 4096]

RESULTS_DIR = Path(__file__).parent / "results"

# ============================================================================
# Helpers
# ============================================================================

def make_mistral_model(max_seq_len, pass_args=None):
    fat._compiled_flex_attention = None

    config = MistralConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=max_seq_len,
        sliding_window=WINDOW_SIZE, # Native SWA enabled
        vocab_size=VOCAB_SIZE,
        _attn_implementation="sdpa",
    )
    model = MistralForCausalLM(config).to(DTYPE).to(DEVICE)

    if pass_args is not None:
        model, stats = flex_attention_transform_pass(model, pass_args)

    model.eval()
    return model

def benchmark_latency(model, seq_len, warmup_iters, bench_iters):
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    latencies = []
    with torch.no_grad():
        for _ in range(bench_iters):
            start_evt.record()
            _ = model(input_ids)
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))

    mean_ms = sum(latencies) / len(latencies)
    std_ms = (sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5
    return {"mean_ms": mean_ms, "std_ms": std_ms, "min_ms": min(latencies),
            "max_ms": max(latencies)}

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 4: Mistral Architecture Integration")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Mistral (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Native Sliding Window: {WINDOW_SIZE}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print()

    configs = [
        ("Native Mistral SWA (SDPA)", None),
        ("Flex Mistral SWA", {"score_mod": "sliding_window", "score_mod_kwargs": {"window_size": WINDOW_SIZE}}),
    ]

    results = {
        "experiment": "exp4_mistral_swa",
        "benchmarks": {},
    }

    for label, pass_args in configs:
        print("-" * 70)
        print(f"Benchmarking: {label}")
        print("-" * 70)
        results["benchmarks"][label] = {}

        for seq_len in SEQ_LENGTHS:
            model = make_mistral_model(seq_len, pass_args)
            torch.cuda.reset_peak_memory_stats()
            print(f"  seq_len={seq_len:5d} ... ", end="", flush=True)

            try:
                timing = benchmark_latency(model, seq_len, WARMUP_ITERS, BENCH_ITERS)
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"mean={timing['mean_ms']:8.2f}ms  mem={peak_mem:.0f}MB")

                results["benchmarks"][label][str(seq_len)] = {
                    "mean_ms": timing["mean_ms"],
                    "peak_memory_mb": peak_mem,
                }
            except torch.cuda.OutOfMemoryError:
                print("OOM — skipped")
                results["benchmarks"][label][str(seq_len)] = {"mean_ms": None, "error": "OOM"}

            del model
            torch.cuda.empty_cache()
        print()

    # --- Summary ---
    col_w = 30
    print("=" * 70)
    print("SPEEDUP (Native SDPA / Flex):")
    print("=" * 70)
    header = f"{'Config':<{col_w}}"
    for sl in SEQ_LENGTHS: header += f" {sl:>8}"
    print(header)
    print("-" * (col_w + 9 * len(SEQ_LENGTHS)))

    row = f"{'Flex Mistral SWA':<{col_w}}"
    for sl in SEQ_LENGTHS:
        sdpa_val = results["benchmarks"]["Native Mistral SWA (SDPA)"].get(str(sl), {}).get("mean_ms")
        flex_val = results["benchmarks"]["Flex Mistral SWA"].get(str(sl), {}).get("mean_ms")
        if sdpa_val and flex_val:
            row += f" {sdpa_val / flex_val:>7.2f}x"
        else:
            row += f" {'N/A':>8}"
    print(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp4_mistral_swa.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()