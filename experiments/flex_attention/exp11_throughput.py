"""
Experiment 11: Throughput Benchmark (Tokens/Second)

Measures tokens/second for both inference and training across sequence
lengths. More interpretable than raw latency for comparing against
published FlexAttention benchmarks.

Throughput = (batch_size * seq_len) / latency_seconds

Model: Medium Llama (hidden=2048, 16 layers, 16 heads, 4 kv_heads)
Dtype: bfloat16

Usage:
    python -u experiments/flex_attention/exp11_throughput.py
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
WINDOW_SIZE = 256

SEQ_LENGTHS = [256, 512, 1024, 2048, 4096]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# SDPA manual masks
# ============================================================================

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


def benchmark_inference(model, seq_len, attention_mask=None):
    """Measure inference (forward-only) latency."""
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)
    fwd_kwargs = {}
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask.expand(BATCH_SIZE, -1, -1, -1)

    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(WARMUP_ITERS):
            _ = model(input_ids, **fwd_kwargs)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    latencies = []

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(BENCH_ITERS):
            start_evt.record()
            _ = model(input_ids, **fwd_kwargs)
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))

    return sum(latencies) / len(latencies)


def benchmark_training(model, seq_len, attention_mask=None):
    """Measure training (forward + backward) latency."""
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)
    labels = input_ids.clone()
    fwd_kwargs = {"labels": labels}
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask.expand(BATCH_SIZE, -1, -1, -1)

    model.train()
    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(WARMUP_ITERS):
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, **fwd_kwargs)
            outputs.loss.backward()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
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

    return sum(latencies) / len(latencies)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 11: Throughput Benchmark (Tokens/Second)")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Dtype: {DTYPE}, Batch Size: {BATCH_SIZE}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
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
        "experiment": "exp11_throughput",
        "config": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV_HEADS,
            "batch_size": BATCH_SIZE, "dtype": str(DTYPE),
            "window_size": WINDOW_SIZE,
            "device": torch.cuda.get_device_name(),
        },
        "throughput": {mode: {} for mode in modes},
    }

    for mode in modes:
        print("=" * 70)
        print(f"MODE: {mode.upper()}")
        print("=" * 70)

        bench_fn = benchmark_inference if mode == "inference" else benchmark_training

        for label, attn_type, pass_args, mask_type in configs:
            print(f"\n  {label}")
            print("  " + "-" * 50)
            results["throughput"][mode][label] = {}

            for seq_len in SEQ_LENGTHS:
                torch.cuda.reset_peak_memory_stats()
                print(f"    seq_len={seq_len:5d} ... ", end="", flush=True)

                model = make_model(seq_len, pass_args if attn_type == "flex" else None)

                attn_mask = None
                if mask_type == "sliding_window":
                    attn_mask = create_sliding_window_mask(seq_len, WINDOW_SIZE, DTYPE, DEVICE)

                try:
                    latency_ms = bench_fn(model, seq_len, attention_mask=attn_mask)
                    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

                    tokens_per_step = BATCH_SIZE * seq_len
                    throughput = tokens_per_step / (latency_ms / 1000)

                    print(
                        f"latency={latency_ms:8.2f}ms  "
                        f"throughput={throughput:10.0f} tok/s  "
                        f"mem={peak_mem:.0f}MB"
                    )

                    results["throughput"][mode][label][str(seq_len)] = {
                        "latency_ms": latency_ms,
                        "throughput_tok_per_sec": throughput,
                        "peak_mem_mb": peak_mem,
                    }
                except torch.cuda.OutOfMemoryError:
                    print("OOM")
                    results["throughput"][mode][label][str(seq_len)] = {"error": "OOM"}
                except RuntimeError as e:
                    print(f"Error ({str(e).splitlines()[0]})")
                    results["throughput"][mode][label][str(seq_len)] = {"error": str(e).splitlines()[0]}

                del model
                if attn_mask is not None:
                    del attn_mask
                torch.cuda.empty_cache()

        print()

    # --- Summary ---
    col_w = 28
    for mode in modes:
        print("=" * 70)
        print(f"SUMMARY: {mode.upper()} Throughput (tokens/sec)")
        print("=" * 70)

        header = f"{'Config':<{col_w}}"
        for sl in SEQ_LENGTHS:
            header += f" {sl:>10}"
        print(header)
        print("-" * (col_w + 11 * len(SEQ_LENGTHS)))

        for label, *_ in configs:
            row = f"{label:<{col_w}}"
            for sl in SEQ_LENGTHS:
                val = results["throughput"][mode][label].get(str(sl), {}).get("throughput_tok_per_sec")
                row += f" {val:>10.0f}" if val is not None else f" {'N/A':>10}"
            print(row)

        # Speedup
        print()
        print(f"THROUGHPUT SPEEDUP (Flex / SDPA) — {mode}:")
        print("-" * (col_w + 11 * len(SEQ_LENGTHS)))

        pairs = [
            ("Flex causal", "SDPA causal"),
            ("Flex sliding_window(256)", "SDPA sliding_window(256)"),
        ]
        for flex_label, sdpa_label in pairs:
            row = f"{flex_label:<{col_w}}"
            for sl in SEQ_LENGTHS:
                flex_val = results["throughput"][mode].get(flex_label, {}).get(str(sl), {}).get("throughput_tok_per_sec")
                sdpa_val = results["throughput"][mode].get(sdpa_label, {}).get(str(sl), {}).get("throughput_tok_per_sec")
                if flex_val and sdpa_val:
                    row += f" {flex_val / sdpa_val:>9.2f}x"
                else:
                    row += f" {'N/A':>10}"
            print(row)

        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp11_throughput.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
