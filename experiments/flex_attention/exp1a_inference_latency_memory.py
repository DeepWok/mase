"""
Experiment 1: Latency Comparison — FlexAttention vs SDPA

Fair comparison across two attention patterns:
  - Causal: SDPA native vs Flex + block_mask
  - Sliding window: SDPA + manual attn_mask vs Flex + block_mask

Model: Medium Llama (hidden=2048, 16 layers, 16 heads, 4 kv_heads) ~1.3B params
Dtype: bfloat16, forward pass only (no training)

Usage:
    python -u experiments/flex_attention/exp1a_inference_latency_memory.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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

# Medium Llama (~1.3B-like)
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

SEQ_LENGTHS = [256, 512, 1024, 2048, 4096]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# SDPA with manual sliding window mask
# ============================================================================

def create_causal_mask(seq_len, dtype, device):
    """
    Create a 4D causal attention mask.

    Format: 0.0 = attend, min_dtype = masked out.
    Shape: (1, 1, seq_len, seq_len) — broadcast over batch and heads.
    """
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), min_val, dtype=dtype, device=device)
    q_idx = torch.arange(seq_len, device=device).view(-1, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, -1)
    mask[0, 0, q_idx >= kv_idx] = 0.0
    return mask.contiguous()


def create_sliding_window_mask(seq_len, window_size, dtype, device):
    """
    Create a 4D sliding window attention mask (causal + window).

    Format: 0.0 = attend, min_dtype = masked out.
    Shape: (1, 1, seq_len, seq_len) — broadcast over batch and heads.
    """
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
    """Create model, optionally with FlexAttention."""
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
        model, stats = flex_attention_transform_pass(model, pass_args)

    model.eval()
    return model


def benchmark_latency(model, seq_len, warmup_iters, bench_iters, attention_mask=None):
    """Measure inference latency in ms using CUDA events."""
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)

    fwd_kwargs = {}
    if attention_mask is not None:
        # Expand to batch size
        fwd_kwargs["attention_mask"] = attention_mask.expand(BATCH_SIZE, -1, -1, -1)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids, **fwd_kwargs)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    latencies = []
    with torch.no_grad():
        for _ in range(bench_iters):
            start_evt.record()
            _ = model(input_ids, **fwd_kwargs)
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))

    mean_ms = sum(latencies) / len(latencies)
    std_ms = (sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5
    return {"mean_ms": mean_ms, "std_ms": std_ms, "min_ms": min(latencies),
            "max_ms": max(latencies), "all_ms": latencies}


def get_peak_memory_mb():
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 1: Latency Comparison — FlexAttention vs SDPA")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, "
          f"heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS})")
    print(f"Dtype: {DTYPE}, Batch size: {BATCH_SIZE}")
    print(f"Warmup: {WARMUP_ITERS}, Bench iters: {BENCH_ITERS}")
    print(f"Sliding window size: {WINDOW_SIZE}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print(f"GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Define configs: (label, type, pass_args, mask_type)
    # mask_type: None = no mask, "causal" = causal mask, "sliding_window" = sliding window mask
    configs = [
        ("SDPA causal",              "sdpa",  None, None),
        ("SDPA sliding_window(256)", "sdpa",  None, "sliding_window"),
        ("Flex causal",              "flex",  {"score_mod": "causal"}, None),
        ("Flex sliding_window(256)", "flex",  {"score_mod": "sliding_window",
                                               "score_mod_kwargs": {"window_size": WINDOW_SIZE}}, None),
    ]

    results = {
        "experiment": "exp1_latency_comparison",
        "config": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV_HEADS,
            "batch_size": BATCH_SIZE, "dtype": str(DTYPE),
            "window_size": WINDOW_SIZE,
            "device": torch.cuda.get_device_name(),
        },
        "benchmarks": {},
    }

    for label, attn_type, pass_args, mask_type in configs:
        print("-" * 70)
        print(f"Benchmarking: {label}")
        print("-" * 70)
        results["benchmarks"][label] = {}

        for seq_len in SEQ_LENGTHS:
            model = make_model(seq_len, pass_args if attn_type == "flex" else None)

            # Create manual mask for SDPA configs
            attn_mask = None
            if mask_type == "causal":
                attn_mask = create_causal_mask(seq_len, DTYPE, DEVICE)
            elif mask_type == "sliding_window":
                attn_mask = create_sliding_window_mask(seq_len, WINDOW_SIZE, DTYPE, DEVICE)

            torch.cuda.reset_peak_memory_stats()
            print(f"  seq_len={seq_len:5d} ... ", end="", flush=True)

            try:
                timing = benchmark_latency(model, seq_len, WARMUP_ITERS, BENCH_ITERS,
                                           attention_mask=attn_mask)
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

                print(f"mean={timing['mean_ms']:8.2f}ms  "
                      f"std={timing['std_ms']:6.2f}ms  "
                      f"mem={peak_mem:.0f}MB")

                results["benchmarks"][label][str(seq_len)] = {
                    "mean_ms": timing["mean_ms"],
                    "std_ms": timing["std_ms"],
                    "min_ms": timing["min_ms"],
                    "max_ms": timing["max_ms"],
                    "peak_memory_mb": peak_mem,
                }
            except torch.cuda.OutOfMemoryError:
                print("OOM — skipped")
                results["benchmarks"][label][str(seq_len)] = {"mean_ms": None, "error": "OOM"}

            del model
            if attn_mask is not None:
                del attn_mask
            torch.cuda.empty_cache()

        print()

    # --- Summary ---
    _print_summary(results, configs, SEQ_LENGTHS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp1a_inference_latency_memory.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _print_summary(results, configs, seq_lengths):
    col_w = 30
    print("=" * 70)
    print("SUMMARY: Latency (mean ms)")
    print("=" * 70)

    header = f"{'Config':<{col_w}}"
    for sl in seq_lengths:
        header += f" {sl:>8}"
    print(header)
    print("-" * (col_w + 9 * len(seq_lengths)))

    for label, *_ in configs:
        row = f"{label:<{col_w}}"
        for sl in seq_lengths:
            entry = results["benchmarks"][label].get(str(sl), {})
            val = entry.get("mean_ms")
            row += f" {val:>8.2f}" if val is not None else f" {'OOM':>8}"
        print(row)

    print()
    print("SUMMARY: Peak Memory (MB)")
    print("-" * (col_w + 9 * len(seq_lengths)))
    for label, *_ in configs:
        row = f"{label:<{col_w}}"
        for sl in seq_lengths:
            entry = results["benchmarks"][label].get(str(sl), {})
            val = entry.get("peak_memory_mb")
            row += f" {val:>8.0f}" if val is not None else f" {'OOM':>8}"
        print(row)

    # Speedup: Flex vs SDPA for each pattern
    print()
    print("SPEEDUP (SDPA / Flex):")
    print("-" * (col_w + 9 * len(seq_lengths)))

    pairs = [
        ("Flex causal", "SDPA causal"),
        ("Flex sliding_window(256)", "SDPA sliding_window(256)"),
    ]
    for flex_label, sdpa_label in pairs:
        row = f"{flex_label:<{col_w}}"
        for sl in seq_lengths:
            sdpa_val = results["benchmarks"].get(sdpa_label, {}).get(str(sl), {}).get("mean_ms")
            flex_val = results["benchmarks"].get(flex_label, {}).get(str(sl), {}).get("mean_ms")
            if sdpa_val and flex_val:
                row += f" {sdpa_val / flex_val:>7.2f}x"
            else:
                row += f" {'N/A':>8}"
        print(row)


if __name__ == "__main__":
    main()
