"""
Kernel Launch Analysis — Part 3 Experiment

Uses torch.profiler to count CUDA kernel dispatches per fusion strategy,
documenting how fused_rmsnorm and flex_attention reduce kernel launch overhead.

Strategies:
  baseline       — raw model, no passes
  fused_rmsnorm  — fused add+RMSNorm Triton kernel (Part 2)
  flex_attention — FlexAttention compiled kernel (Part 1)
  both           — fused_rmsnorm + flex_attention combined

Usage:
    python scripts/profile_kernels.py --model tinyllama --save-dir outputs/profiling/
    python scripts/profile_kernels.py --model mistral   --save-dir outputs/profiling/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoModelForCausalLM

from chop.passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)
from chop.passes.module.transforms.fused_ops.rmsnorm_residual_fusion import (
    fused_rmsnorm_residual_transform_pass,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "tinyllama": {
        "checkpoint": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "load_dtype": torch.float16,
        "score_mod": "causal",
        "score_mod_kwargs": {},
        "vocab_size": 32000,
        "num_layers": 22,
    },
    "mistral": {
        "checkpoint": "mistralai/Mistral-7B-v0.1",
        "load_dtype": torch.float16,
        "score_mod": "sliding_window",
        "score_mod_kwargs": {"window_size": 128},
        "vocab_size": 32000,
        "num_layers": 32,
    },
}

STRATEGIES = ["baseline", "fused_rmsnorm", "flex_attention", "both"]

# Warmup runs before profiling.  flex_attention uses torch.compile so needs
# enough passes for Triton/inductor compilation to finish before we measure.
_WARMUP = {
    "baseline": 3,
    "fused_rmsnorm": 3,
    "flex_attention": 10,   # compile on first ~3 runs
    "both": 10,
}


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(strategy: str, cfg: dict, device: str):
    """Load a fresh copy of the model from disk for each strategy.

    Avoids deepcopy which requires a full second copy of the model in RAM —
    fatal for large models like Mistral-7B (~14 GB × 2 = 28 GB peak).
    """
    model = AutoModelForCausalLM.from_pretrained(
        cfg["checkpoint"], torch_dtype=cfg["load_dtype"]
    ).to(device).eval()

    if strategy in ("fused_rmsnorm", "both"):
        model, _ = fused_rmsnorm_residual_transform_pass(model, {})

    if strategy in ("flex_attention", "both"):
        model, _ = flex_attention_transform_pass(
            model,
            {
                "score_mod": cfg["score_mod"],
                "score_mod_kwargs": cfg["score_mod_kwargs"],
            },
        )

    return model


def make_batch(seq_len: int, vocab_size: int, device: str):
    return {
        "input_ids": torch.randint(0, vocab_size, (1, seq_len), device=device),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_strategy(model, batch, num_warmup: int):
    """
    Returns kernel statistics dict for one forward pass captured by
    torch.profiler.

    Requires both CPU + CUDA activities: CPU tracing provides the op-level
    call counts and kernel launch attribution; CUDA tracing measures GPU
    execution time.  CUDA-attributed events are those with cuda_time_total > 0.
    """
    with torch.no_grad():
        for _ in range(num_warmup):
            model(**batch)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with record_function("model_forward"):
            with torch.no_grad():
                model(**batch)
        torch.cuda.synchronize()

    avgs = prof.key_averages()

    # self_device_time_total > 0: ops that directly dispatched GPU kernels
    # (device_time_total includes parent ops that contain GPU children — avoid
    #  double-counting by using the self variant)
    cuda_events = [e for e in avgs if e.self_device_time_total > 0]

    total_launches = sum(e.count for e in cuda_events)
    unique_kernels = len(cuda_events)
    total_cuda_ms = sum(e.self_device_time_total for e in cuda_events) / 1e3  # µs→ms

    # RMSNorm / fused-kernel events by name
    norm_events = [
        e for e in cuda_events
        if any(kw in e.key.lower() for kw in ("norm", "rms", "fused", "triton"))
    ]
    norm_launches = sum(e.count for e in norm_events)

    # Top-5 kernels by GPU time
    top5 = sorted(cuda_events, key=lambda e: e.self_device_time_total, reverse=True)[:5]

    return {
        "total_kernel_launches": total_launches,
        "unique_kernels": unique_kernels,
        "total_cuda_time_ms": round(total_cuda_ms, 3),
        "norm_related_launches": norm_launches,
        "top5_kernels": [
            {
                "name": e.key[:70],
                "count": e.count,
                "cuda_time_ms": round(e.self_device_time_total / 1e3, 3),
            }
            for e in top5
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), default="tinyllama")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/profiling"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available — results will not reflect GPU kernel counts.")

    cfg = MODEL_CONFIGS[args.model]
    args.save_dir.mkdir(parents=True, exist_ok=True)

    batch = make_batch(args.seq_len, cfg["vocab_size"], device)

    results = {
        "model": args.model,
        "checkpoint": cfg["checkpoint"],
        "seq_len": args.seq_len,
        "num_layers": cfg["num_layers"],
        "dtype": str(cfg["load_dtype"]),
        "strategies": {},
    }

    header = f"\n{'Strategy':<18} {'Total launches':>16} {'Unique kernels':>15} {'CUDA time (ms)':>15} {'Norm launches':>14}"
    print(header)
    print("-" * len(header))

    for strategy in STRATEGIES:
        print(f"  Building {strategy} ...", flush=True)
        model = None
        try:
            model = build_model(strategy, cfg, device)
            stats = profile_strategy(model, batch, num_warmup=_WARMUP[strategy])
            results["strategies"][strategy] = stats

            print(
                f"  {strategy:<16} "
                f"{stats['total_kernel_launches']:>16,} "
                f"{stats['unique_kernels']:>15,} "
                f"{stats['total_cuda_time_ms']:>15.2f} "
                f"{stats['norm_related_launches']:>14,}"
            )
        except Exception as e:
            print(f"  [ERROR] {strategy}: {e}")
            results["strategies"][strategy] = {"error": str(e)}
        finally:
            if model is not None:
                model.to("cpu")
                del model
            torch.cuda.empty_cache()

    # Reduction summary vs baseline
    baseline = results["strategies"].get("baseline", {})
    b = baseline.get("total_kernel_launches", 0)
    if b > 0:
        print("\n  Reduction vs baseline:")
        for s in STRATEGIES[1:]:
            s_stats = results["strategies"].get(s, {})
            if "total_kernel_launches" in s_stats:
                reduction = b - s_stats["total_kernel_launches"]
                pct = 100 * reduction / b
                print(f"    {s:<18} {reduction:>+6} launches  ({pct:.1f}% vs baseline)")

    out_path = args.save_dir / f"kernel_profile_{args.model}_seq{args.seq_len}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
