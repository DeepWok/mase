"""
Large-model latency benchmark: FlexAttention (causal) vs SDPA.

Uses a Llama-7B-like config (hidden=4096, 32 layers, 32 heads, 8 kv_heads)
in bfloat16 on a single GPU. Forward pass only, no training.

Usage:
    python -u experiments/flex_attention/latency_benchmark_large.py

Output:
    - Printed latency table
    - JSON results saved to experiments/flex_attention/results/latency_benchmark_large.json
"""

import json
import sys
import time
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

# Llama-7B-like config
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 11008
NUM_LAYERS = 32
NUM_HEADS = 32
NUM_KV_HEADS = 8
VOCAB_SIZE = 32000
BATCH_SIZE = 1
WARMUP_ITERS = 3
BENCH_ITERS = 10
DEVICE = "cuda"
DTYPE = torch.bfloat16

SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

# Only 2 configs: SDPA vs Flex causal (with block_mask)
CONFIGS = [
    ("SDPA (eager)", "sdpa", None),
    ("Flex causal", "flex", {"score_mod": "causal"}),
]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# Helpers
# ============================================================================

def make_model(max_seq_len, pass_args=None):
    """Create a Llama-7B-like model, optionally with FlexAttention."""
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
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)

    if pass_args is not None:
        model, stats = flex_attention_transform_pass(model, pass_args)

    model.eval()
    return model


def benchmark_latency(model, seq_len, warmup_iters, bench_iters):
    """Measure average inference latency in milliseconds using CUDA events."""
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids)
    torch.cuda.synchronize()

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


def get_gpu_memory_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("FlexAttention vs SDPA — Large Model Latency Benchmark")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Llama-7B-like (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, "
          f"heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS})")
    print(f"Dtype: {DTYPE}, Batch size: {BATCH_SIZE}")
    print(f"Warmup: {WARMUP_ITERS}, Bench iters: {BENCH_ITERS}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    results = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "vocab_size": VOCAB_SIZE,
            "batch_size": BATCH_SIZE,
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
            "dtype": str(DTYPE),
            "device": torch.cuda.get_device_name(),
        },
        "benchmarks": {},
    }

    for label, attn_type, pass_args in CONFIGS:
        print("-" * 70)
        print(f"Benchmarking: {label}")
        print("-" * 70)

        results["benchmarks"][label] = {}

        for seq_len in SEQ_LENGTHS:
            model = make_model(seq_len, pass_args if attn_type == "flex" else None)
            mem_mb = get_gpu_memory_mb()

            print(f"  seq_len={seq_len:5d} (GPU mem: {mem_mb:.0f} MB) ... ",
                  end="", flush=True)

            try:
                timing = benchmark_latency(model, seq_len, WARMUP_ITERS, BENCH_ITERS)
                print(f"mean={timing['mean_ms']:8.2f}ms  "
                      f"std={timing['std_ms']:6.2f}ms  "
                      f"min={timing['min_ms']:8.2f}ms")

                results["benchmarks"][label][str(seq_len)] = {
                    "mean_ms": timing["mean_ms"],
                    "std_ms": timing["std_ms"],
                    "min_ms": timing["min_ms"],
                    "max_ms": timing["max_ms"],
                    "gpu_mem_mb": mem_mb,
                }
            except torch.cuda.OutOfMemoryError:
                print("OOM — skipped")
                results["benchmarks"][label][str(seq_len)] = {"mean_ms": None, "error": "OOM"}

            del model
            torch.cuda.empty_cache()

        print()

    # --- Summary table ---
    print("=" * 70)
    print("SUMMARY TABLE (mean latency in ms)")
    print("=" * 70)

    header = f"{'Config':<25}"
    for sl in SEQ_LENGTHS:
        header += f" {sl:>8}"
    print(header)
    print("-" * (25 + 9 * len(SEQ_LENGTHS)))

    sdpa_key = CONFIGS[0][0]
    sdpa_data = results["benchmarks"][sdpa_key]

    for label, _, _ in CONFIGS:
        row = f"{label:<25}"
        for sl in SEQ_LENGTHS:
            entry = results["benchmarks"][label].get(str(sl), {})
            val = entry.get("mean_ms")
            row += f" {val:>8.2f}" if val is not None else f" {'OOM':>8}"
        print(row)

    print()
    print("Speedup (SDPA / Flex):")
    print("-" * (25 + 9 * len(SEQ_LENGTHS)))

    for label, attn_type, _ in CONFIGS:
        if attn_type == "sdpa":
            continue
        row = f"{label:<25}"
        for sl in SEQ_LENGTHS:
            sdpa_val = sdpa_data.get(str(sl), {}).get("mean_ms")
            flex_val = results["benchmarks"][label].get(str(sl), {}).get("mean_ms")
            if sdpa_val and flex_val:
                speedup = sdpa_val / flex_val
                row += f" {speedup:>7.2f}x"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # --- Save ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "latency_benchmark_large.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
