"""
Quick sanity check: is sliding_window with block_mask fast or catastrophically slow?

Only tests 3 configs at seq_len=1024 with minimal iterations.
Should finish in ~5 minutes (mostly torch.compile warmup).

PASS criteria: sliding_window latency < 50ms (was 7000ms+ before fix)

Usage:
    python -u experiments/flex_attention/quick_sanity.py
"""

import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chop.passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)
import chop.passes.module.transforms.attention.flex_attention_transform as fat

DEVICE = "cuda"
SEQ_LEN = 1024
BATCH_SIZE = 4
WARMUP = 3
BENCH = 5

CONFIGS = [
    ("SDPA (eager)", None),
    ("Flex causal (block_mask)", {"score_mod": "causal"}),
    ("Flex sliding_window(128)", {"score_mod": "sliding_window", "score_mod_kwargs": {"window_size": 128}}),
]


def make_model(pass_args=None):
    fat._compiled_flex_attention = None
    config = LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=SEQ_LEN, vocab_size=512,
        _attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).to(DEVICE)
    if pass_args is not None:
        model, _ = flex_attention_transform_pass(model, pass_args)
    model.eval()
    return model


def bench(model):
    input_ids = torch.randint(0, 512, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_ids)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(BENCH):
            start.record()
            model(input_ids)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def main():
    print("=" * 60)
    print("Quick Sanity Check: block_mask fix verification")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"seq_len={SEQ_LEN}, batch={BATCH_SIZE}, warmup={WARMUP}, bench={BENCH}")
    print("=" * 60)

    all_pass = True
    for label, pass_args in CONFIGS:
        print(f"\n{label} ... ", end="", flush=True)
        model = make_model(pass_args)
        ms = bench(model)
        del model
        torch.cuda.empty_cache()

        status = "OK" if ms < 50 else "SLOW"
        if ms >= 50:
            all_pass = False
        print(f"{ms:.2f} ms  [{status}]")

    print("\n" + "=" * 60)
    if all_pass:
        print("PASS: All configs < 50ms. Block mask fix is working.")
    else:
        print("FAIL: Some configs still slow. Block mask is NOT working.")
    print("=" * 60)


if __name__ == "__main__":
    main()
