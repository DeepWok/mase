"""
Experiment 6: Compound Masks (ALiBi + Sliding Window)

Proves that FlexAttention can fuse complex mathematical biases (ALiBi) 
and sparsity masks (Sliding Window) into a single Triton kernel with 
virtually zero latency penalty.

Model: Medium Llama (hidden=2048, 16 layers, 16 heads, 4 kv_heads)
Dtype: bfloat16, Forward + Backward
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

SEQ_LENGTHS = [256, 1024, 4096]

RESULTS_DIR = Path(__file__).parent / "results"

def make_model(max_seq_len, pass_args=None):
    fat._compiled_flex_attention = None
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS, max_position_embeddings=max_seq_len,
        vocab_size=VOCAB_SIZE, _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)
    if pass_args is not None:
        model, _ = flex_attention_transform_pass(model, pass_args)
    model.train()
    return model

def benchmark_latency(model, seq_len):
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)
    labels = input_ids.clone()

    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(WARMUP_ITERS):
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, labels=labels)
            outputs.loss.backward()
            
    torch.cuda.synchronize()
    start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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
    return {"mean_ms": mean_ms, "peak_mem_mb": torch.cuda.max_memory_allocated() / 1024 / 1024}

def main():
    print("=" * 70)
    print("Experiment 6: Compound Masks (ALiBi + Sliding Window)")
    print("=" * 70)

    configs = [
        ("Flex Causal", {"score_mod": "causal"}),
        ("Flex Sliding Window", {
            "score_mod": "sliding_window", 
            "score_mod_kwargs": {"window_size": WINDOW_SIZE}
        }),
        ("Flex ALiBi + SWA", {
            "score_mod": "alibi_sliding_window", 
            "score_mod_kwargs": {"num_heads": NUM_HEADS, "window_size": WINDOW_SIZE},
            "mask_mod": "sliding_window",
            "mask_mod_kwargs": {"window_size": WINDOW_SIZE}
        }),
    ]

    results = {"experiment": "exp6_compound_masks", "benchmarks": {}}

    for label, pass_args in configs:
        print(f"\nBenchmarking: {label}")
        print("-" * 50)
        results["benchmarks"][label] = {}

        for seq_len in SEQ_LENGTHS:
            torch.cuda.reset_peak_memory_stats()
            print(f"  seq_len={seq_len:4d} ... ", end="", flush=True)
            model = make_model(seq_len, pass_args)
            try:
                timing = benchmark_latency(model, seq_len)
                print(f"mean={timing['mean_ms']:8.2f}ms  mem={timing['peak_mem_mb']:.0f}MB")
                results["benchmarks"][label][str(seq_len)] = timing
            except torch.cuda.OutOfMemoryError:
                print("OOM")
            del model
            torch.cuda.empty_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp6_compound_masks.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()