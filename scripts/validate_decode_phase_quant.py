#!/usr/bin/env python3
"""Validate decode-only phase-split quantisation (Disaggregated serving)

1) GPTQ calibrates linear weights with ``phase="decode"`` — the GPTQ result
   becomes the DECODE weight bank while prefill stays unquantised
2) Phase-aware quantized modules are installed through
   ``quantize_module_transform_pass`` with decode-only configs
   (``prefill: {bypass: True}``)
3) KV cache follows the handoff rule: prefill KV writes are stored in the
   decode chip's MX format (``kv_cache_handoff="decode_format"``).
4) The script prints a generation preview plus a decode-phase perplexity
   readout (``force_runtime_phase("decode")``)

Works for Llama and Qwen3 checkpoints (Dense).
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chop.passes.module.transforms.gptq.data_utils import get_loaders
from chop.passes.module.transforms.quantize.quantize import (
    quantize_module_transform_pass,
)
from chop.passes.module.transforms.quantize.rotation_search import (
    _compute_calibration_perplexity,
)


@dataclass(frozen=True)
class ValidationConfig:
    model_name: str
    device: str
    dtype: str
    local_files_only: bool
    trust_remote_code: bool
    prompt: str
    max_new_tokens: int
    seed: int
    print_chars: int
    # decode-side quantisation settings
    weight_width: int
    weight_block_size: int
    act_width: int
    act_block_size: int
    kv_width: int
    kv_block_size: int
    use_gptq: bool
    # GPTQ settings
    gptq_dataset: str
    gptq_nsamples: int
    gptq_seqlen: int
    gptq_cali_batch_size: int
    gptq_max_layers: int | None
    gptq_checkpoint_dir: str | None
    hf_token: str | None
    # eval
    eval_ppl_nsamples: int
    eval_ppl_seqlen: int


def _parse_args() -> ValidationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Validate decode-only phase-split quantisation: FP prefill, "
            "MXINT decode weights/activations/KV, optional GPTQ (phase=decode)."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id (Llama or Qwen3 dense).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--local_files_only",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    parser.add_argument(
        "--trust_remote_code",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "You are a helpful assistant. Briefly explain three practical "
            "steps to debug slow Python code, then end with one-line advice."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_chars", type=int, default=1200)

    # Decode-side mixed-precision knobs
    parser.add_argument("--weight_width", type=int, default=4)
    parser.add_argument("--weight_block_size", type=int, default=32)
    parser.add_argument("--act_width", type=int, default=8)
    parser.add_argument("--act_block_size", type=int, default=32)
    parser.add_argument("--kv_width", type=int, default=4)
    parser.add_argument("--kv_block_size", type=int, default=32)
    parser.add_argument(
        "--use_gptq",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="GPTQ-calibrate the decode weight bank (phase='decode'). "
        "False = RTN decode weights.",
    )

    # GPTQ
    parser.add_argument(
        "--gptq_dataset",
        type=str,
        default="wikitext2",
        choices=("wikitext2", "c4", "ptb"),
    )
    parser.add_argument("--gptq_nsamples", type=int, default=128)
    parser.add_argument("--gptq_seqlen", type=int, default=2048)
    parser.add_argument("--gptq_cali_batch_size", type=int, default=32)
    parser.add_argument("--gptq_max_layers", type=int, default=None)
    parser.add_argument("--gptq_checkpoint_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)

    # decode-phase perplexity readout
    parser.add_argument("--eval_ppl_nsamples", type=int, default=16)
    parser.add_argument("--eval_ppl_seqlen", type=int, default=1024)

    args = parser.parse_args()
    return ValidationConfig(**vars(args))


def _resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_decode_only_pass_args(cfg: ValidationConfig) -> dict[str, Any]:
    """Decode-only quantisation config: FP prefill chip, MXINT decode chip.

    Weights/activations/KV each carry independent precisions
    """

    linear_decode = {
        "weight_block_size": cfg.weight_block_size,
        "weight_width": cfg.weight_width,
        "data_in_block_size": cfg.act_block_size,
        "data_in_width": cfg.act_width,
    }
    if cfg.use_gptq:
        # GPTQ already produced the decode bank; don't RTN-requantise it.
        linear_decode = {"gptq": True, **linear_decode}

    attn_decode = {
        # Keep qk/av/softmax/rope FP here so HF's fast attention backend
        # remains usable; the memory-wall wins come from W + KV. Flip these
        # to quantised configs (with _attn_implementation="eager") to model
        # the in-attention arithmetic too.
        "qk_matmul": {"bypass": True},
        "av_matmul": {"bypass": True},
        "rope": {"bypass": True},
        "softmax": {"bypass": True},
        "kv_cache": {
            "data_in_block_size": cfg.kv_block_size,
            "data_in_width": cfg.kv_width,
        },
    }

    pass_args: dict[str, Any] = {
        "by": "regex_name",
        r"model\.layers\.\d+\.self_attn$": {
            "config": {
                "name": "mxint",
                # kv_cache_handoff defaults to "decode_format": prefill KV
                # is quantised on write into the decode chip's HBM format.
                "prefill": {"bypass": True},
                "decode": attn_decode,
            }
        },
        r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        r"|mlp\.(gate_proj|up_proj|down_proj))$": {
            "config": {
                "name": "mxint",
                "prefill": {"bypass": True},
                "decode": linear_decode,
            }
        },
    }

    if cfg.use_gptq:
        pass_args["gptq"] = {
            "model_name": cfg.model_name,
            "device": cfg.device,
            "dataset": cfg.gptq_dataset,
            "nsamples": cfg.gptq_nsamples,
            "seqlen": cfg.gptq_seqlen,
            "format": "mxint",
            "weight_config": {
                "weight_block_size": cfg.weight_block_size,
                "weight_width": cfg.weight_width,
            },
            # Decode-side flow: GPTQ output becomes the decode weight bank,
            # FP weights are restored so prefill stays unquantised.
            "phase": "decode",
            "quantile_search": True,
            "clip_search_y": False,
            "cali_batch_size": cfg.gptq_cali_batch_size,
            "checkpoint_dir": cfg.gptq_checkpoint_dir,
            "hf_token": cfg.hf_token,
            "max_layers": cfg.gptq_max_layers,
        }

    return pass_args


def _load_model_and_tokenizer(cfg: ValidationConfig):
    load_kwargs = {
        "local_files_only": cfg.local_files_only,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.hf_token:
        load_kwargs["token"] = cfg.hf_token
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=_resolve_torch_dtype(cfg.dtype),
        **load_kwargs,
    )
    model = model.to(cfg.device)
    model.eval()
    return model, tokenizer


# Perplexity accounting is shared with the rotation-search pass
# (_compute_calibration_perplexity), so this readout stays comparable with
# the numbers the DSE flow reports. Its score_phase argument forces every
# token through the requested chip's numerics.


@torch.no_grad()
def _generate_preview(model, tokenizer, cfg: ValidationConfig) -> str:
    inputs = tokenizer(cfg.prompt, return_tensors="pt").to(cfg.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    cfg = _parse_args()
    _set_deterministic_seed(cfg.seed)

    print("=" * 88)
    print("Decode-only phase-split quantisation validator (PLENA)")
    print(f"model_name       : {cfg.model_name}")
    print(f"device / dtype   : {cfg.device} / {cfg.dtype}")
    print(f"W / A / KV width : {cfg.weight_width} / {cfg.act_width} / {cfg.kv_width} (MXINT)")
    print(f"use_gptq (decode): {cfg.use_gptq}")
    print("=" * 88)

    print("[1/5] Loading model and tokenizer...")
    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer(cfg)
    print(f"      done in {time.time() - t0:.2f}s")

    print("[2/5] Building decode-only quantisation config...")
    pass_args = build_decode_only_pass_args(cfg)

    print("[3/5] Running quantize_module_transform_pass "
          f"({'incl. GPTQ phase=decode' if cfg.use_gptq else 'RTN decode banks'})...")
    t1 = time.time()
    model, _ = quantize_module_transform_pass(model, pass_args)
    model = model.to(cfg.device)
    model.eval()
    print(f"      done in {time.time() - t1:.2f}s")

    print("[4/5] Perplexity readout (prefill=FP chip vs decode=quantised chip)...")
    loader = get_loaders(
        cfg.gptq_dataset,
        nsamples=cfg.eval_ppl_nsamples,
        seed=0,
        seqlen=cfg.eval_ppl_seqlen,
        model=cfg.model_name,
        hf_token=cfg.hf_token,
    )
    ppl_prefill = _compute_calibration_perplexity(
        model, loader, cfg.device, label="prefill_fp_chip", score_phase="prefill"
    )
    ppl_decode = _compute_calibration_perplexity(
        model, loader, cfg.device, label="decode_quantised_chip", score_phase="decode"
    )
    print(f"      ppl[prefill numerics / FP chip]      : {ppl_prefill:.4f}")
    print(f"      ppl[decode numerics / quantised chip]: {ppl_decode:.4f}")

    print("[5/5] Generation preview (real prefill->decode phase switching)...")
    out = _generate_preview(model, tokenizer, cfg)

    print("\n" + "=" * 88)
    print("Prompt:")
    print(cfg.prompt)
    print("-" * 88)
    print("Generated text preview:")
    print(out[: cfg.print_chars] if cfg.print_chars > 0 else out)
    if cfg.print_chars > 0 and len(out) > cfg.print_chars:
        print(f"\n... [truncated to {cfg.print_chars} chars]")
    print("=" * 88)


if __name__ == "__main__":
    # Example:
    # CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/validate_decode_phase_quant.py \
    #     --model_name Qwen/Qwen3-8B --device cuda:0
    main()
