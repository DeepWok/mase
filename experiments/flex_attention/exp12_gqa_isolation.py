"""
Experiment 12: GQA vs MHA Isolation

Isolates the impact of Grouped Query Attention on FlexAttention performance.
The transform pass sets enable_gqa=True when num_kv_heads < num_heads, but
this has never been benchmarked in isolation.

Compares three head configurations (all with num_attention_heads=16):
  - MHA:   num_kv_heads=16 (no grouping)
  - GQA-4: num_kv_heads=4  (4x grouping, the default in other experiments)
  - MQA:   num_kv_heads=1  (maximum grouping)

For each, compares SDPA causal vs Flex causal (training mode).

Model: Medium Llama (hidden=2048, 16 layers)
Dtype: bfloat16, Forward + Backward

Usage:
    python -u experiments/flex_attention/exp12_gqa_isolation.py
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
VOCAB_SIZE = 32000
BATCH_SIZE = 2
WARMUP_ITERS = 3
BENCH_ITERS = 10
DEVICE = "cuda"
DTYPE = torch.bfloat16

HEAD_CONFIGS = [
    ("MHA (16/16)",  16),  # num_kv_heads = num_heads -> no GQA
    ("GQA-4 (16/4)",  4),  # 4x grouping (default in other experiments)
    ("MQA (16/1)",     1),  # maximum grouping
]

SEQ_LENGTHS = [512, 1024, 2048, 4096]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# Helpers
# ============================================================================

def make_model(max_seq_len, num_kv_heads, use_flex=False):
    fat._compiled_flex_attention = None

    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=max_seq_len,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)

    if use_flex:
        model, _ = flex_attention_transform_pass(model, {"score_mod": "causal"})

    model.train()
    return model


def benchmark_training(model, seq_len):
    """Measure training (forward + backward) latency."""
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)
    labels = input_ids.clone()

    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(WARMUP_ITERS):
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, labels=labels)
            outputs.loss.backward()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    latencies = []

    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(BENCH_ITERS):
            model.zero_grad(set_to_none=True)
            start_evt.record()
            outputs = model(input_ids, labels=labels)
            outputs.loss.backward()
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))

    mean_ms = sum(latencies) / len(latencies)
    return mean_ms


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 12: GQA vs MHA Isolation")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Num attention heads: {NUM_HEADS}")
    print(f"Dtype: {DTYPE}, Batch Size: {BATCH_SIZE}")
    print(f"Head configs: {[(n, kv) for n, kv in HEAD_CONFIGS]}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print()

    results = {
        "experiment": "exp12_gqa_isolation",
        "config": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "batch_size": BATCH_SIZE,
            "dtype": str(DTYPE),
            "device": torch.cuda.get_device_name(),
        },
        "benchmarks": {},
    }

    for head_label, num_kv_heads in HEAD_CONFIGS:
        print("=" * 70)
        print(f"HEAD CONFIG: {head_label} (num_kv_heads={num_kv_heads})")
        print("=" * 70)

        for attn_label, use_flex in [("SDPA", False), ("Flex", True)]:
            full_label = f"{attn_label} {head_label}"
            print(f"\n  {full_label}")
            print("  " + "-" * 50)
            results["benchmarks"][full_label] = {}

            for seq_len in SEQ_LENGTHS:
                torch.cuda.reset_peak_memory_stats()
                print(f"    seq_len={seq_len:5d} ... ", end="", flush=True)

                model = make_model(seq_len, num_kv_heads, use_flex=use_flex)

                try:
                    latency_ms = benchmark_training(model, seq_len)
                    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

                    print(f"mean={latency_ms:8.2f}ms  mem={peak_mem:.0f}MB")

                    results["benchmarks"][full_label][str(seq_len)] = {
                        "mean_ms": latency_ms,
                        "peak_mem_mb": peak_mem,
                    }
                except torch.cuda.OutOfMemoryError:
                    print("OOM")
                    results["benchmarks"][full_label][str(seq_len)] = {"error": "OOM"}
                except RuntimeError as e:
                    print(f"Error ({str(e).splitlines()[0]})")
                    results["benchmarks"][full_label][str(seq_len)] = {"error": str(e).splitlines()[0]}

                del model
                torch.cuda.empty_cache()

        print()

    # --- Summary ---
    col_w = 22
    print("=" * 70)
    print("SUMMARY: Training Latency (ms)")
    print("=" * 70)

    header = f"{'Config':<{col_w}}"
    for sl in SEQ_LENGTHS:
        header += f" {sl:>10}"
    print(header)
    print("-" * (col_w + 11 * len(SEQ_LENGTHS)))

    for head_label, _ in HEAD_CONFIGS:
        for attn_label in ["SDPA", "Flex"]:
            full_label = f"{attn_label} {head_label}"
            row = f"{full_label:<{col_w}}"
            for sl in SEQ_LENGTHS:
                val = results["benchmarks"][full_label].get(str(sl), {}).get("mean_ms")
                row += f" {val:>10.2f}" if val is not None else f" {'N/A':>10}"
            print(row)
        print()  # Visual separator between head configs

    # Speedup per head config
    print("SPEEDUP (SDPA / Flex) per head config:")
    print("-" * (col_w + 11 * len(SEQ_LENGTHS)))

    for head_label, _ in HEAD_CONFIGS:
        sdpa_label = f"SDPA {head_label}"
        flex_label = f"Flex {head_label}"
        row = f"{head_label:<{col_w}}"
        for sl in SEQ_LENGTHS:
            sdpa_val = results["benchmarks"].get(sdpa_label, {}).get(str(sl), {}).get("mean_ms")
            flex_val = results["benchmarks"].get(flex_label, {}).get(str(sl), {}).get("mean_ms")
            if sdpa_val and flex_val:
                row += f" {sdpa_val / flex_val:>9.2f}x"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # Memory comparison
    print()
    print("MEMORY: Peak Memory (MB)")
    print("-" * (col_w + 11 * len(SEQ_LENGTHS)))

    for head_label, _ in HEAD_CONFIGS:
        for attn_label in ["SDPA", "Flex"]:
            full_label = f"{attn_label} {head_label}"
            row = f"{full_label:<{col_w}}"
            for sl in SEQ_LENGTHS:
                val = results["benchmarks"][full_label].get(str(sl), {}).get("peak_mem_mb")
                row += f" {val:>10.0f}" if val is not None else f" {'N/A':>10}"
            print(row)
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp12_gqa_isolation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
