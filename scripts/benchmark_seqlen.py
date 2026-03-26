"""
Scaling benchmarks — Experiments 3, 4, 5 from the project plan.

  Exp 3: Latency vs seq_len [64→4096] — log-log shows sub-quadratic FlexAttention scaling
  Exp 4: Latency + throughput vs batch_size [1→16] at fixed seq_len=512
  Exp 5: Peak GPU memory vs seq_len — FlexAttention O(n*w) vs SDPA O(n²)

Five configs per experiment:
  baseline      — raw model, no passes (FP32 for TinyLlama, FP16 for Mistral)
  int8_none     — INT8 quantisation, standard SDPA
  int8_flex     — INT8 quantisation + FlexAttention (causal / sliding_window)
  int8_rmsnorm  — INT8 quantisation + fused RMSNorm residual (Part 2)
  int8_both     — INT8 quantisation + FlexAttention + fused RMSNorm (full stack)

At seq_len=512 these five rows form the complete ablation table (Experiment 2).

Usage:
    python scripts/benchmark_seqlen.py --model tinyllama --save-dir outputs/benchmark/
    python scripts/benchmark_seqlen.py --model mistral   --save-dir outputs/benchmark/
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer

from chop.passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)
from chop.passes.module.transforms.quantize.quantize import (
    quantize_module_transform_pass,
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "tinyllama": {
        "checkpoint": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "score_mod": "causal",
        "score_mod_kwargs": {},
        "load_dtype": torch.float32,
        "vocab_size": 32000,
        # Exp 3+5: seq_len sweep at batch=1
        "seq_lens": [64, 128, 256, 512, 1024, 2048, 4096],
        "seqlen_batch_size": 1,
        # Exp 4: batch sweep at seq_len=512
        "batch_sizes": [1, 4, 8, 16],
        "batchsize_seq_len": 512,
    },
    "mistral": {
        "checkpoint": "mistralai/Mistral-7B-v0.1",
        "score_mod": "sliding_window",
        "score_mod_kwargs": {"window_size": 128},
        # FP16: float32 Mistral-7B (~28 GB) OOMs on L40S 40 GB
        "load_dtype": torch.float16,
        "vocab_size": 32000,
        # Exp 3+5: stop at 2048 — standard SDPA OOMs above this for 7B
        "seq_lens": [64, 128, 256, 512, 1024, 2048],
        "seqlen_batch_size": 1,
        # Exp 4: only batch=1 for 7B model
        "batch_sizes": [1],
        "batchsize_seq_len": 512,
    },
}

_QUANT_INT8 = {
    "by": "type",
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: str):
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
    }


def build_model(base_model, strategy: str, score_mod: str, score_mod_kwargs: dict, device: str):
    """
    Deep-copy base_model and apply passes for the given strategy.

      'baseline'       — no passes, original dtype
      'int8_none'      — INT8 quantisation only
      'int8_flex'      — INT8 quantisation + FlexAttention
      'int8_rmsnorm'   — INT8 quantisation + fused RMSNorm (Part 2)
      'int8_both'      — INT8 quantisation + FlexAttention + fused RMSNorm (Part 2)
    """
    model = deepcopy(base_model).to(device)
    orig_dtype = next(model.parameters()).dtype

    if strategy == "baseline":
        model.eval()
        return model

    # INT8 quantisation — always applied for non-baseline configs.
    # deepcopy the config: quantize_module_transform_pass mutates its args dict.
    model, _ = quantize_module_transform_pass(model, deepcopy(_QUANT_INT8))
    # Restore original dtype (quantize pass creates float32 LinearInteger modules)
    model = model.to(device=device, dtype=orig_dtype)

    if strategy in ("int8_flex", "int8_both"):
        model, _ = flex_attention_transform_pass(
            model,
            {"score_mod": score_mod, "score_mod_kwargs": score_mod_kwargs},
        )

    if strategy in ("int8_rmsnorm", "int8_both"):
        from chop.passes.module.transforms.fused_ops.rmsnorm_residual_fusion import (
            rmsnorm_residual_fusion_pass,
        )
        model, _ = rmsnorm_residual_fusion_pass(model, {})

    model.eval()
    return model


def time_forward(model, batch, num_warmup: int, num_batches: int) -> float:
    """Returns mean latency in ms over num_batches timed forward passes."""
    with torch.no_grad():
        for _ in range(num_warmup):
            model(**batch)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        for _ in range(num_batches):
            model(**batch)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_batches


def peak_memory_mb(model, batch) -> float:
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(**batch)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def run_safely(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), return None on OOM."""
    try:
        return fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("    [OOM]")
        return None


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

STRATEGIES = ["baseline", "int8_none", "int8_flex", "int8_rmsnorm", "int8_both"]


def exp_seqlen(base_model, cfg, device, num_warmup, num_batches):
    """Experiments 3 + 5: latency and peak memory vs sequence length."""
    results = {}
    for seq_len in cfg["seq_lens"]:
        results[seq_len] = {}
        batch = make_batch(cfg["seqlen_batch_size"], seq_len, cfg["vocab_size"], device)

        for strategy in STRATEGIES:
            print(f"  [seqlen] seq_len={seq_len:5d}  strategy={strategy}")
            model = None
            try:
                model = build_model(
                    base_model, strategy,
                    cfg["score_mod"], cfg["score_mod_kwargs"], device,
                )
                latency = run_safely(time_forward, model, batch, num_warmup, num_batches)
                memory  = run_safely(peak_memory_mb, model, batch)
                results[seq_len][strategy] = {
                    "latency_ms": latency,
                    "peak_memory_mb": memory,
                }
            except Exception as e:
                print(f"    [ERROR] {e}")
                results[seq_len][strategy] = {"latency_ms": None, "peak_memory_mb": None}
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()

    return results


def exp_batchsize(base_model, cfg, device, num_warmup, num_batches):
    """Experiment 4: latency and throughput vs batch size at fixed seq_len."""
    results = {}
    for batch_size in cfg["batch_sizes"]:
        results[batch_size] = {}
        batch = make_batch(batch_size, cfg["batchsize_seq_len"], cfg["vocab_size"], device)

        for strategy in STRATEGIES:
            print(f"  [batch]  batch_size={batch_size:3d}  strategy={strategy}")
            model = None
            try:
                model = build_model(
                    base_model, strategy,
                    cfg["score_mod"], cfg["score_mod_kwargs"], device,
                )
                latency = run_safely(time_forward, model, batch, num_warmup, num_batches)
                throughput = (batch_size * 1000 / latency) if latency else None
                results[batch_size][strategy] = {
                    "latency_ms": latency,
                    "throughput_samples_per_sec": throughput,
                }
            except Exception as e:
                print(f"    [ERROR] {e}")
                results[batch_size][strategy] = {
                    "latency_ms": None,
                    "throughput_samples_per_sec": None,
                }
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--num-warmup",  type=int, default=5)
    parser.add_argument("--num-batches", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MODEL_CONFIGS[args.model]
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg['checkpoint']} in {cfg['load_dtype']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["checkpoint"], torch_dtype=cfg["load_dtype"]
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n=== Exp 3+5: Sequence Length Scaling ({args.model}) ===")
    seqlen_results = exp_seqlen(model, cfg, device, args.num_warmup, args.num_batches)

    print(f"\n=== Exp 4: Batch Size Scaling ({args.model}) ===")
    batch_results = exp_batchsize(model, cfg, device, args.num_warmup, args.num_batches)

    output = {
        "model": args.model,
        "checkpoint": cfg["checkpoint"],
        "score_mod": cfg["score_mod"],
        "score_mod_kwargs": cfg["score_mod_kwargs"],
        "num_warmup": args.num_warmup,
        "num_batches": args.num_batches,
        "seqlen_scaling": seqlen_results,
        "batchsize_scaling": batch_results,
    }

    out_path = args.save_dir / f"benchmark_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
