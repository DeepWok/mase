"""
Experiment 3: Block Mask Ablation

Quantifies the impact of block_mask on FlexAttention latency.
Compares with vs without block_mask for causal and sliding_window patterns.

Model: Medium Llama (hidden=2048, 16 layers, 16 heads, 4 kv_heads)
Dtype: bfloat16, forward pass only

Usage:
    python -u experiments/flex_attention/exp3_block_mask_ablation.py
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
torch._dynamo.config.cache_size_limit = 64

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
WINDOW_SIZE = 256

# Pushed to 4096 to clearly demonstrate quadratic divergence
SEQ_LENGTHS = [256, 512, 1024, 2048, 4096]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# Helpers
# ============================================================================

def make_model(max_seq_len, pass_args):
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
    model, _ = flex_attention_transform_pass(model, pass_args)
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
    print("Experiment 3: Block Mask Ablation")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Dtype: {DTYPE}, Batch size: {BATCH_SIZE}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print()

    configs = [
        ("Causal + block_mask",         {"score_mod": "causal"}),
        ("Causal (no block_mask)",       {"score_mod": "causal", "use_block_mask": False}),
        ("Sliding_window + block_mask",  {"score_mod": "sliding_window",
                                          "score_mod_kwargs": {"window_size": WINDOW_SIZE}}),
        ("Sliding_window (no block_mask)", {"score_mod": "sliding_window",
                                            "score_mod_kwargs": {"window_size": WINDOW_SIZE},
                                            "use_block_mask": False}),
    ]

    results = {
        "experiment": "exp3_block_mask_ablation",
        "config": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV_HEADS,
            "batch_size": BATCH_SIZE, "dtype": str(DTYPE),
            "window_size": WINDOW_SIZE,
            "device": torch.cuda.get_device_name(),
        },
        "benchmarks": {},
    }

    for label, pass_args in configs:
        print("-" * 70)
        print(f"Benchmarking: {label}")
        print("-" * 70)
        results["benchmarks"][label] = {}

        for seq_len in SEQ_LENGTHS:
            model = make_model(seq_len, pass_args)
            print(f"  seq_len={seq_len:5d} ... ", end="", flush=True)

            try:
                timing = benchmark_latency(model, seq_len, WARMUP_ITERS, BENCH_ITERS)
                print(f"mean={timing['mean_ms']:8.2f}ms  std={timing['std_ms']:6.2f}ms")

                results["benchmarks"][label][str(seq_len)] = {
                    "mean_ms": timing["mean_ms"],
                    "std_ms": timing["std_ms"],
                    "min_ms": timing["min_ms"],
                    "max_ms": timing["max_ms"],
                }
            except torch.cuda.OutOfMemoryError:
                print("OOM — skipped")
                results["benchmarks"][label][str(seq_len)] = {"mean_ms": None, "error": "OOM"}

            del model
            torch.cuda.empty_cache()
        print()

    # --- Summary ---
    col_w = 35
    print("=" * 70)
    print("SUMMARY: Latency (mean ms)")
    print("=" * 70)

    header = f"{'Config':<{col_w}}"
    for sl in SEQ_LENGTHS:
        header += f" {sl:>8}"
    print(header)
    print("-" * (col_w + 9 * len(SEQ_LENGTHS)))

    for label, _ in configs:
        row = f"{label:<{col_w}}"
        for sl in SEQ_LENGTHS:
            entry = results["benchmarks"][label].get(str(sl), {})
            val = entry.get("mean_ms")
            row += f" {val:>8.2f}" if val is not None else f" {'OOM':>8}"
        print(row)

    # Speedup: with vs without block_mask
    print()
    print("SPEEDUP (no_block_mask / with_block_mask):")
    print("-" * (col_w + 9 * len(SEQ_LENGTHS)))

    pairs = [
        ("Causal", "Causal + block_mask", "Causal (no block_mask)"),
        ("Sliding_window", "Sliding_window + block_mask", "Sliding_window (no block_mask)"),
    ]
    for pair_label, with_label, without_label in pairs:
        row = f"{pair_label:<{col_w}}"
        for sl in SEQ_LENGTHS:
            with_val = results["benchmarks"].get(with_label, {}).get(str(sl), {}).get("mean_ms")
            without_val = results["benchmarks"].get(without_label, {}).get(str(sl), {}).get("mean_ms")
            if with_val and without_val:
                row += f" {without_val / with_val:>7.2f}x"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # --- Save ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp3_block_mask_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()