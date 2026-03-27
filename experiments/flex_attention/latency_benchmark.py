"""
Latency benchmark: FlexAttention vs SDPA at various sequence lengths and score_mods.

Measures inference latency (ms) for:
  - SDPA baseline (eager attention)
  - FlexAttention with causal score_mod (with block_mask)
  - FlexAttention with causal score_mod (without block_mask, for comparison)
  - FlexAttention with sliding_window score_mods (with block_mask)

Varies sequence length: 128, 256, 512, 1024, 2048

Usage:
    python experiments/flex_attention/latency_benchmark.py

Output:
    - Printed latency table
    - JSON results saved to experiments/flex_attention/results/latency_benchmark.json
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chop.passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)
import chop.passes.module.transforms.attention.flex_attention_transform as fat


# ============================================================================
# Config
# ============================================================================

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
VOCAB_SIZE = 512
BATCH_SIZE = 4
WARMUP_ITERS = 5
BENCH_ITERS = 20
DEVICE = "cuda"
DTYPE = torch.float32

SEQ_LENGTHS = [128, 256, 512, 1024, 2048]

# Configurations to benchmark: (label, attention_type, pass_args_or_None)
# attention_type: "sdpa" = baseline, "flex" = FlexAttention
CONFIGS = [
    ("SDPA (eager)", "sdpa", None),
    ("Flex causal", "flex", {"score_mod": "causal"}),
    ("Flex causal (no block_mask)", "flex", {"score_mod": "causal", "use_block_mask": False}),
    ("Flex sliding_window(128)", "flex", {
        "score_mod": "sliding_window",
        "score_mod_kwargs": {"window_size": 128},
    }),
    ("Flex sliding_window(256)", "flex", {
        "score_mod": "sliding_window",
        "score_mod_kwargs": {"window_size": 256},
    }),
]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# Helpers
# ============================================================================

def make_model(max_seq_len, pass_args=None):
    """Create a tiny Llama model, optionally with FlexAttention."""
    # Reset compiled flex_attention cache for each new config
    fat._compiled_flex_attention = None

    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=max_seq_len,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).to(DEVICE).to(DTYPE)

    if pass_args is not None:
        model, stats = flex_attention_transform_pass(model, pass_args)

    model.eval()
    return model


def benchmark_latency(model, seq_len, warmup_iters, bench_iters):
    """Measure average inference latency in milliseconds using CUDA events."""
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)

    # Warmup (includes torch.compile first-call cost)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    with torch.no_grad():
        for _ in range(bench_iters):
            start_event.record()
            _ = model(input_ids)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

    mean_ms = sum(latencies) / len(latencies)
    std_ms = (sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5
    min_ms = min(latencies)
    max_ms = max(latencies)

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "all_ms": latencies,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("FlexAttention vs SDPA Latency Benchmark")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: tiny Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, "
          f"heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS})")
    print(f"Batch size: {BATCH_SIZE}, Warmup: {WARMUP_ITERS}, "
          f"Bench iters: {BENCH_ITERS}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print()

    results = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "vocab_size": VOCAB_SIZE,
            "batch_size": BATCH_SIZE,
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
            "device": torch.cuda.get_device_name(),
            "dtype": str(DTYPE),
        },
        "benchmarks": {},
    }

    # Run benchmarks
    for label, attn_type, pass_args in CONFIGS:
        print("-" * 70)
        print(f"Benchmarking: {label}")
        print("-" * 70)

        results["benchmarks"][label] = {}

        for seq_len in SEQ_LENGTHS:
            # Need max_position_embeddings >= seq_len
            model = make_model(seq_len, pass_args if attn_type == "flex" else None)

            print(f"  seq_len={seq_len:5d} ... ", end="", flush=True)
            timing = benchmark_latency(model, seq_len, WARMUP_ITERS, BENCH_ITERS)

            print(f"mean={timing['mean_ms']:8.2f}ms  "
                  f"std={timing['std_ms']:6.2f}ms  "
                  f"min={timing['min_ms']:8.2f}ms")

            results["benchmarks"][label][str(seq_len)] = {
                "mean_ms": timing["mean_ms"],
                "std_ms": timing["std_ms"],
                "min_ms": timing["min_ms"],
                "max_ms": timing["max_ms"],
            }

            # Free model memory between runs
            del model
            torch.cuda.empty_cache()

        print()

    # --- Summary table ---
    print("=" * 70)
    print("SUMMARY TABLE (mean latency in ms)")
    print("=" * 70)

    # Header
    header = f"{'Config':<35}"
    for sl in SEQ_LENGTHS:
        header += f" {sl:>8}"
    print(header)
    print("-" * (35 + 9 * len(SEQ_LENGTHS)))

    # Get SDPA baseline for speedup calculation
    sdpa_key = CONFIGS[0][0]
    sdpa_data = results["benchmarks"][sdpa_key]

    for label, _, _ in CONFIGS:
        row = f"{label:<35}"
        for sl in SEQ_LENGTHS:
            val = results["benchmarks"][label][str(sl)]["mean_ms"]
            row += f" {val:>8.2f}"
        print(row)

    # Speedup rows
    print()
    print("Speedup vs SDPA:")
    print("-" * (35 + 9 * len(SEQ_LENGTHS)))

    for label, attn_type, _ in CONFIGS:
        if attn_type == "sdpa":
            continue
        row = f"{label:<35}"
        for sl in SEQ_LENGTHS:
            sdpa_val = sdpa_data[str(sl)]["mean_ms"]
            flex_val = results["benchmarks"][label][str(sl)]["mean_ms"]
            speedup = sdpa_val / flex_val if flex_val > 0 else float("inf")
            row += f" {speedup:>7.2f}x"
        print(row)

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "latency_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
