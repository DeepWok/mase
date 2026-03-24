"""
Experiment 12: GQA vs MHA Isolation (Enhanced with SWA + ALiBi+SWA)

Isolates the impact of Grouped Query Attention on FlexAttention performance
across multiple attention patterns. Tests a 3x3 matrix:

Head configs (all with num_attention_heads=16):
  - MHA:   num_kv_heads=16 (no grouping)
  - GQA-4: num_kv_heads=4  (4x grouping)
  - MQA:   num_kv_heads=1  (maximum grouping)

Attention patterns:
  - Causal (baseline — FA2 is optimized for this)
  - Sliding Window (256) — block sparsity skips distant blocks
  - ALiBi + Sliding Window (256) — compound score_mod with SWA block_mask

For each cell, compares SDPA vs Flex (training mode, forward + backward).

The key hypothesis: Flex's advantage over SDPA comes from block sparsity
(SWA/ALiBi+SWA), and this advantage is independent of the GQA grouping
ratio. The official PyTorch paper showed GQA+ALiBi decode at 5.37x — this
experiment tests whether the GQA × attention_pattern interaction holds
during training.

Model: Medium Llama (hidden=2048, 16 layers)
Dtype: bfloat16, Forward + Backward

Usage:
    python -u experiments/flex_attention/exp12_gqa_isolation.py
"""

import copy
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
WINDOW_SIZE = 256

HEAD_CONFIGS = [
    ("MHA (16/16)",  16),  # num_kv_heads = num_heads -> no GQA
    ("GQA-4 (16/4)",  4),  # 4x grouping (default in other experiments)
    ("MQA (16/1)",     1),  # maximum grouping
]

# Attention patterns: (label, flex_pass_args, sdpa_needs_manual_mask)
# For ALiBi+SWA, num_heads in score_mod_kwargs is filled per head_config
ATTN_PATTERNS = [
    ("causal", {"score_mod": "causal"}, False),
    ("SWA(256)", {
        "score_mod": "sliding_window",
        "score_mod_kwargs": {"window_size": WINDOW_SIZE},
    }, True),
    ("ALiBi+SWA(256)", {
        "score_mod": "alibi_sliding_window",
        "score_mod_kwargs": {"num_heads": None, "window_size": WINDOW_SIZE},
        "mask_mod": "sliding_window",
        "mask_mod_kwargs": {"window_size": WINDOW_SIZE},
    }, True),
]

SEQ_LENGTHS = [512, 1024, 2048, 4096]

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# SDPA manual masks
# ============================================================================

def create_sliding_window_mask(seq_len, window_size, dtype, device):
    """Create a 4D attention mask for sliding window causal attention (SDPA fallback)."""
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

def make_model(num_kv_heads, pass_args=None):
    """Build model once with max seq len. Reuse across shorter sequences."""
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=max(SEQ_LENGTHS),
        vocab_size=VOCAB_SIZE,
        _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)

    if pass_args is not None:
        model, _ = flex_attention_transform_pass(model, pass_args)

    model.train()
    return model


def benchmark_training(model, seq_len, attention_mask=None):
    """Measure training (forward + backward) latency."""
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

    mean_ms = sum(latencies) / len(latencies)
    return mean_ms


def prepare_flex_pass_args(attn_pass_args, num_heads):
    """Deep copy pass_args and fill in num_heads for ALiBi variants."""
    args = copy.deepcopy(attn_pass_args)
    kwargs = args.get("score_mod_kwargs", {})
    if "num_heads" in kwargs and kwargs["num_heads"] is None:
        kwargs["num_heads"] = num_heads
    return args


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 12: GQA vs MHA Isolation (Enhanced)")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Medium Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Num attention heads: {NUM_HEADS}")
    print(f"Dtype: {DTYPE}, Batch Size: {BATCH_SIZE}")
    print(f"Head configs: {[(n, kv) for n, kv in HEAD_CONFIGS]}")
    print(f"Attention patterns: {[p[0] for p in ATTN_PATTERNS]}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print()

    results = {
        "experiment": "exp12_gqa_isolation",
        "config": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "batch_size": BATCH_SIZE,
            "dtype": str(DTYPE), "window_size": WINDOW_SIZE,
            "device": torch.cuda.get_device_name(),
        },
        "benchmarks": {},
    }

    for head_label, num_kv_heads in HEAD_CONFIGS:
        print("=" * 70)
        print(f"HEAD CONFIG: {head_label} (num_kv_heads={num_kv_heads})")
        print("=" * 70)

        for attn_label, attn_pass_args, needs_manual_mask in ATTN_PATTERNS:
            print(f"\n  --- Attention Pattern: {attn_label} ---")

            for backend in ["SDPA", "Flex"]:
                full_label = f"{backend} {head_label} {attn_label}"
                print(f"\n  {full_label}")
                print("  " + "-" * 50)
                results["benchmarks"][full_label] = {}

                # Build model ONCE per (head_config, attn_pattern, backend)
                # Reuse across all seq_lengths — only input tensor size changes
                try:
                    if backend == "Flex":
                        fat._compiled_flex_attention = None  # reset cache for new pattern
                        pass_args = prepare_flex_pass_args(attn_pass_args, NUM_HEADS)
                        model = make_model(num_kv_heads, pass_args=pass_args)
                    else:
                        model = make_model(num_kv_heads, pass_args=None)
                except Exception as e:
                    print(f"    Model creation failed: {str(e).splitlines()[0]}")
                    for seq_len in SEQ_LENGTHS:
                        results["benchmarks"][full_label][str(seq_len)] = {"error": str(e).splitlines()[0]}
                    continue

                for seq_len in SEQ_LENGTHS:
                    torch.cuda.reset_peak_memory_stats()
                    print(f"    seq_len={seq_len:5d} ... ", end="", flush=True)

                    try:
                        attn_mask = None
                        if backend == "SDPA" and needs_manual_mask:
                            attn_mask = create_sliding_window_mask(
                                seq_len, WINDOW_SIZE, DTYPE, DEVICE
                            )

                        latency_ms = benchmark_training(model, seq_len, attention_mask=attn_mask)
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

                    if attn_mask is not None:
                        del attn_mask
                    torch.cuda.empty_cache()

                del model
                torch.cuda.empty_cache()

        print()

    # --- Summary Tables ---
    col_w = 36
    print("\n" + "=" * 70)
    print("SUMMARY: Training Latency (ms)")
    print("=" * 70)

    header = f"{'Config':<{col_w}}"
    for sl in SEQ_LENGTHS:
        header += f" {sl:>10}"
    print(header)
    print("-" * (col_w + 11 * len(SEQ_LENGTHS)))

    for head_label, _ in HEAD_CONFIGS:
        for attn_label, _, _ in ATTN_PATTERNS:
            for backend in ["SDPA", "Flex"]:
                full_label = f"{backend} {head_label} {attn_label}"
                row = f"{full_label:<{col_w}}"
                for sl in SEQ_LENGTHS:
                    val = results["benchmarks"].get(full_label, {}).get(str(sl), {}).get("mean_ms")
                    row += f" {val:>10.2f}" if val is not None else f" {'N/A':>10}"
                print(row)
            print()  # separator between SDPA/Flex pairs

    # --- Speedup Matrix ---
    print("=" * 70)
    print("SPEEDUP MATRIX: (SDPA / Flex) — higher = Flex is faster")
    print("=" * 70)
    print()

    # Print a nice matrix: rows = head_configs, columns = attn_patterns, for each seq_len
    for sl in SEQ_LENGTHS:
        print(f"  Seq Length: {sl}")
        matrix_col_w = 18
        header = f"  {'Head Config':<{matrix_col_w}}"
        for attn_label, _, _ in ATTN_PATTERNS:
            header += f" {attn_label:>18}"
        print(header)
        print("  " + "-" * (matrix_col_w + 19 * len(ATTN_PATTERNS)))

        for head_label, _ in HEAD_CONFIGS:
            row = f"  {head_label:<{matrix_col_w}}"
            for attn_label, _, _ in ATTN_PATTERNS:
                sdpa_key = f"SDPA {head_label} {attn_label}"
                flex_key = f"Flex {head_label} {attn_label}"
                sdpa_val = results["benchmarks"].get(sdpa_key, {}).get(str(sl), {}).get("mean_ms")
                flex_val = results["benchmarks"].get(flex_key, {}).get(str(sl), {}).get("mean_ms")
                if sdpa_val and flex_val:
                    row += f" {sdpa_val / flex_val:>17.2f}x"
                else:
                    row += f" {'N/A':>18}"
            print(row)
        print()

    # --- Memory Matrix ---
    print("=" * 70)
    print("MEMORY: Peak Memory (MB) at seq_len=4096")
    print("=" * 70)

    mem_col_w = 36
    header = f"{'Config':<{mem_col_w}} {'Memory (MB)':>12}"
    print(header)
    print("-" * (mem_col_w + 13))

    for head_label, _ in HEAD_CONFIGS:
        for attn_label, _, _ in ATTN_PATTERNS:
            for backend in ["SDPA", "Flex"]:
                full_label = f"{backend} {head_label} {attn_label}"
                val = results["benchmarks"].get(full_label, {}).get("4096", {}).get("peak_mem_mb")
                if val is not None:
                    print(f"{full_label:<{mem_col_w}} {val:>12.0f}")
                else:
                    print(f"{full_label:<{mem_col_w}} {'N/A':>12}")
            print()

    # --- Key Insights ---
    print()
    print("KEY INSIGHTS:")
    print("- Causal: Flex ≈ SDPA (FA2 is heavily optimized for causal)")
    print("- SWA: Flex should be faster due to block sparsity skipping distant blocks")
    print("- ALiBi+SWA: Same block sparsity as SWA, with added score bias — near-zero overhead")
    print("- The Flex speedup should be independent of GQA grouping ratio")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp12_gqa_isolation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
