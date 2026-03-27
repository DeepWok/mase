"""
Experiment 4: Batch Size Sensitivity

Tests whether FlexAttention's speedup over SDPA holds across batch sizes.
Fixed seq_len=1024, varies batch_size=[1, 2, 4, 8].

Compares:
  - SDPA causal
  - Flex causal + block_mask

Model: Medium Llama (hidden=2048, 16 layers, 16 heads, 4 kv_heads)
Dtype: bfloat16, forward pass only

Usage:
    python -u experiments/flex_attention/exp9_batch_sensitivity.py
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

# ============================================================================
# Config
# ============================================================================

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 5504
NUM_LAYERS = 16
NUM_HEADS = 16
NUM_KV_HEADS = 4
VOCAB_SIZE = 32000
SEQ_LEN = 1024
WARMUP_ITERS = 3
BENCH_ITERS = 10
DEVICE = "cuda"
DTYPE = torch.bfloat16

BATCH_SIZES = [1, 2, 4, 8]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# Helpers
# ============================================================================

def make_model(use_flex=False):
    fat._compiled_flex_attention = None

    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="sdpa" if not use_flex else "eager",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)

    if use_flex:
        model, _ = flex_attention_transform_pass(model, {"score_mod": "causal"})

    model.eval()
    return model


def benchmark_latency(model, batch_size, warmup_iters, bench_iters):
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=DEVICE)

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
    print("Experiment 4: Batch Size Sensitivity")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Dtype: {DTYPE}, Seq len: {SEQ_LEN}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print()

    configs = [
        ("SDPA causal", False),
        ("Flex causal", True),
    ]

    results = {
        "experiment": "exp4_batch_sensitivity",
        "config": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV_HEADS,
            "seq_len": SEQ_LEN, "dtype": str(DTYPE),
            "device": torch.cuda.get_device_name(),
        },
        "benchmarks": {},
    }

    for label, use_flex in configs:
        print("-" * 70)
        print(f"Benchmarking: {label}")
        print("-" * 70)
        results["benchmarks"][label] = {}

        for bs in BATCH_SIZES:
            model = make_model(use_flex=use_flex)
            torch.cuda.reset_peak_memory_stats()
            print(f"  batch_size={bs:2d} ... ", end="", flush=True)

            try:
                timing = benchmark_latency(model, bs, WARMUP_ITERS, BENCH_ITERS)
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

                print(f"mean={timing['mean_ms']:8.2f}ms  "
                      f"std={timing['std_ms']:6.2f}ms  "
                      f"mem={peak_mem:.0f}MB")

                results["benchmarks"][label][str(bs)] = {
                    "mean_ms": timing["mean_ms"],
                    "std_ms": timing["std_ms"],
                    "min_ms": timing["min_ms"],
                    "max_ms": timing["max_ms"],
                    "peak_memory_mb": peak_mem,
                }
            except torch.cuda.OutOfMemoryError:
                print("OOM — skipped")
                results["benchmarks"][label][str(bs)] = {"mean_ms": None, "error": "OOM"}

            del model
            torch.cuda.empty_cache()
        print()

    # --- Summary ---
    col_w = 20
    print("=" * 70)
    print(f"SUMMARY (seq_len={SEQ_LEN}, latency in ms)")
    print("=" * 70)

    header = f"{'Config':<{col_w}}"
    for bs in BATCH_SIZES:
        header += f" {'bs=' + str(bs):>10}"
    print(header)
    print("-" * (col_w + 11 * len(BATCH_SIZES)))

    for label, _ in configs:
        row = f"{label:<{col_w}}"
        for bs in BATCH_SIZES:
            val = results["benchmarks"][label].get(str(bs), {}).get("mean_ms")
            row += f" {val:>10.2f}" if val is not None else f" {'OOM':>10}"
        print(row)

    print()
    print("SPEEDUP (SDPA / Flex):")
    print("-" * (col_w + 11 * len(BATCH_SIZES)))
    row = f"{'Flex causal':<{col_w}}"
    for bs in BATCH_SIZES:
        sdpa_val = results["benchmarks"]["SDPA causal"].get(str(bs), {}).get("mean_ms")
        flex_val = results["benchmarks"]["Flex causal"].get(str(bs), {}).get("mean_ms")
        if sdpa_val and flex_val:
            row += f" {sdpa_val / flex_val:>9.2f}x"
        else:
            row += f" {'N/A':>10}"
    print(row)

    # --- Save ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp4_batch_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
