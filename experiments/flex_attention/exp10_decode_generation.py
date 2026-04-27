"""
Experiment 10: Decode/Generation Benchmark

Measures autoregressive generation performance (prefill + decode phases).
Key hypothesis: FlexAttention's block sparsity provides NO benefit during
decode because Q_LEN=1 < FLEX_BLOCK_SIZE=128, so block_mask is always None.

This means FlexAttention helps during prefill (full prompt forward pass)
but not during token-by-token generation with KV cache.

Model: Medium Llama (hidden=2048, 16 layers, 16 heads, 4 kv_heads)
Dtype: bfloat16, Batch Size: 1

Usage:
    python -u experiments/flex_attention/exp10_decode_generation.py
"""

import json
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, GenerationConfig

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
BATCH_SIZE = 1
DEVICE = "cuda"
DTYPE = torch.bfloat16
WINDOW_SIZE = 256

PROMPT_LENGTHS = [128, 512, 1024]
NUM_NEW_TOKENS = 128
WARMUP_RUNS = 2
BENCH_RUNS = 5

RESULTS_DIR = Path(__file__).parent / "results"


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
        max_position_embeddings=max_seq_len + NUM_NEW_TOKENS,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)

    if pass_args is not None:
        model, _ = flex_attention_transform_pass(model, pass_args)

    model.eval()
    return model


def benchmark_prefill(model, prompt_ids):
    """Measure prefill latency (single forward pass over the full prompt)."""
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(WARMUP_RUNS):
            _ = model(prompt_ids)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    latencies = []

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(BENCH_RUNS):
            start_evt.record()
            _ = model(prompt_ids)
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))

    return sum(latencies) / len(latencies)


def benchmark_generation(model, prompt_ids, num_new_tokens):
    """Measure total generation time (prefill + all decode steps)."""
    gen_config = GenerationConfig(
        max_new_tokens=num_new_tokens,
        do_sample=False,  # Greedy for determinism
    )

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(WARMUP_RUNS):
            _ = model.generate(prompt_ids, generation_config=gen_config)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    latencies = []

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(BENCH_RUNS):
            start_evt.record()
            output = model.generate(prompt_ids, generation_config=gen_config)
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))

    total_ms = sum(latencies) / len(latencies)
    tokens_generated = output.shape[1] - prompt_ids.shape[1]
    return total_ms, tokens_generated


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 10: Decode/Generation Benchmark")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Dtype: {DTYPE}, Batch Size: {BATCH_SIZE}")
    print(f"Prompt lengths: {PROMPT_LENGTHS}, New tokens: {NUM_NEW_TOKENS}")
    print(f"Warmup: {WARMUP_RUNS}, Bench runs: {BENCH_RUNS}")
    print()

    configs = [
        ("SDPA causal", None),
        ("Flex causal", {"score_mod": "causal"}),
        ("Flex sliding_window(256)", {
            "score_mod": "sliding_window",
            "score_mod_kwargs": {"window_size": WINDOW_SIZE},
        }),
    ]

    results = {
        "experiment": "exp10_decode_generation",
        "config": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV_HEADS,
            "batch_size": BATCH_SIZE, "dtype": str(DTYPE),
            "num_new_tokens": NUM_NEW_TOKENS,
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

        for prompt_len in PROMPT_LENGTHS:
            torch.cuda.reset_peak_memory_stats()
            print(f"  prompt_len={prompt_len:4d} ... ", end="", flush=True)

            model = make_model(prompt_len, pass_args)
            prompt_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, prompt_len), device=DEVICE)

            try:
                # Phase 1: Prefill only
                prefill_ms = benchmark_prefill(model, prompt_ids)

                # Phase 2: Full generation (prefill + decode)
                total_ms, tokens_generated = benchmark_generation(
                    model, prompt_ids, NUM_NEW_TOKENS,
                )

                # Derive decode metrics
                decode_ms = total_ms - prefill_ms
                per_token_ms = decode_ms / tokens_generated if tokens_generated > 0 else float("inf")
                tokens_per_sec = tokens_generated / (decode_ms / 1000) if decode_ms > 0 else 0

                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

                print(
                    f"prefill={prefill_ms:7.2f}ms  "
                    f"decode={decode_ms:7.2f}ms  "
                    f"per_token={per_token_ms:5.2f}ms  "
                    f"tok/s={tokens_per_sec:7.1f}  "
                    f"mem={peak_mem:.0f}MB"
                )

                results["benchmarks"][label][str(prompt_len)] = {
                    "prefill_ms": prefill_ms,
                    "total_generation_ms": total_ms,
                    "decode_ms": decode_ms,
                    "per_token_ms": per_token_ms,
                    "tokens_per_sec": tokens_per_sec,
                    "tokens_generated": tokens_generated,
                    "peak_mem_mb": peak_mem,
                }
            except torch.cuda.OutOfMemoryError:
                print("OOM")
                results["benchmarks"][label][str(prompt_len)] = {"error": "OOM"}
            except RuntimeError as e:
                print(f"Error ({str(e).splitlines()[0]})")
                results["benchmarks"][label][str(prompt_len)] = {"error": str(e).splitlines()[0]}

            del model
            torch.cuda.empty_cache()

        print()

    # --- Summary ---
    col_w = 28
    print("=" * 70)
    print("SUMMARY: Prefill Latency (ms)")
    print("=" * 70)
    header = f"{'Config':<{col_w}}"
    for pl in PROMPT_LENGTHS:
        header += f" {pl:>10}"
    print(header)
    print("-" * (col_w + 11 * len(PROMPT_LENGTHS)))

    for label, _ in configs:
        row = f"{label:<{col_w}}"
        for pl in PROMPT_LENGTHS:
            val = results["benchmarks"][label].get(str(pl), {}).get("prefill_ms")
            row += f" {val:>10.2f}" if val is not None else f" {'N/A':>10}"
        print(row)

    print()
    print("SUMMARY: Decode Per-Token Latency (ms)")
    print("-" * (col_w + 11 * len(PROMPT_LENGTHS)))
    for label, _ in configs:
        row = f"{label:<{col_w}}"
        for pl in PROMPT_LENGTHS:
            val = results["benchmarks"][label].get(str(pl), {}).get("per_token_ms")
            row += f" {val:>10.2f}" if val is not None else f" {'N/A':>10}"
        print(row)

    print()
    print("SUMMARY: Decode Throughput (tokens/sec)")
    print("-" * (col_w + 11 * len(PROMPT_LENGTHS)))
    for label, _ in configs:
        row = f"{label:<{col_w}}"
        for pl in PROMPT_LENGTHS:
            val = results["benchmarks"][label].get(str(pl), {}).get("tokens_per_sec")
            row += f" {val:>10.1f}" if val is not None else f" {'N/A':>10}"
        print(row)

    print()
    print("KEY INSIGHT: FlexAttention's block_mask is skipped during decode")
    print("(Q_LEN=1 < FLEX_BLOCK_SIZE=128), so decode throughput should be")
    print("similar across SDPA and Flex configs. The speedup is in prefill only.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp10_decode_generation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
