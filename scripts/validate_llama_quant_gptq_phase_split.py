#!/usr/bin/env python3
"""Validate Llama GPTQ + phase-split integration in the module quantize flow.

This script is a *standalone integration validator* for the current step-1
quantization design:
1) GPTQ quantizes Llama linear weights with calibration data (wikitext2).
2) Quantized Llama wrapper modules are installed through
   ``quantize_module_transform_pass``.
3) Runtime phase is inferred by decoder-layer pre-hooks.
4) Decode path stays full precision by policy (``decode_policy=fp_only``).

The script intentionally prioritizes robustness and maintainability over
aggressive compression. It uses conservative quantization settings and prints
clear runtime summaries for human inspection.
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

from chop.passes.module.transforms.quantize.quantize import (
    quantize_module_transform_pass,
)


@dataclass(frozen=True)
class ValidationConfig:
    """Runtime configuration for this integration validator."""

    model_name: str
    device: str
    dtype: str
    local_files_only: bool
    trust_remote_code: bool
    prompt: str
    max_new_tokens: int
    seed: int
    print_chars: int
    # GPTQ settings
    gptq_dataset: str
    gptq_nsamples: int
    gptq_seqlen: int
    gptq_cali_batch_size: int
    gptq_max_layers: int | None
    gptq_checkpoint_dir: str | None
    hf_token: str | None


def _parse_args() -> ValidationConfig:
    """Parse CLI arguments and return an immutable runtime config."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate Llama GPTQ + phase-split module quantization "
            "using local HF cache and conservative settings."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model id. Must exist in local cache when offline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=(
            "Torch device string. Use with CUDA_VISIBLE_DEVICES=1 to pin this "
            "process to physical GPU1."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--local_files_only",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Force loading model/tokenizer/dataset from local cache only.",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Forwarded to HF model/tokenizer loaders.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "You are a helpful assistant. Briefly explain three practical "
            "steps to debug slow Python code, then end with one-line advice."
        ),
        help="Validation prompt used for generation output inspection.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
        help="Maximum generated tokens for the validation output.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    parser.add_argument(
        "--print_chars",
        type=int,
        default=1200,
        help="Print at most this many output characters (human inspection).",
    )

    # GPTQ
    parser.add_argument(
        "--gptq_dataset",
        type=str,
        default="wikitext2",
        choices=("wikitext2", "c4", "ptb"),
        help="Calibration dataset for GPTQ.",
    )
    parser.add_argument("--gptq_nsamples", type=int, default=128)
    parser.add_argument("--gptq_seqlen", type=int, default=256)
    parser.add_argument("--gptq_cali_batch_size", type=int, default=32)
    parser.add_argument(
        "--gptq_max_layers",
        type=int,
        default=None,
        help="Optional limit for GPTQ layers; None means full-layer quantization.",
    )
    parser.add_argument(
        "--gptq_checkpoint_dir",
        type=str,
        default=None,
        help="Optional checkpoint directory for GPTQ resume.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional HF token if your cached model metadata still requires it.",
    )

    args = parser.parse_args()
    return ValidationConfig(**vars(args))


def _resolve_torch_dtype(name: str) -> torch.dtype:
    """Map CLI dtype name to torch dtype."""

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def _set_deterministic_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility of this validator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_conservative_pass_args(cfg: ValidationConfig) -> dict[str, Any]:
    """Build conservative quantization config for module quantization pass.

    Design intent:
    - Keep dynamic quantization paths in attention/MLP/RMS mostly bypassed to
      reduce quality regression risk in this validator.
    - Let GPTQ provide the primary compression effect on linear weights.
    - Preserve phase-structured config fields to validate prefill/decode wiring.
    """

    # Conservative phase sub-config for attention kernels.
    # We keep bypass=True to prioritize text readability while still exercising
    # phase-structured config parsing and hook-based runtime phase selection.
    attn_phase_cfg = {
        "qk_matmul": {"bypass": True},
        "av_matmul": {"bypass": True},
        "rope": {"bypass": True},
        "softmax": {"bypass": True},
        "kv_cache": {"bypass": True},
    }

    # For linear_mxfp:
    # - prefill uses GPTQ weight result (gptq=True prevents re-PTQ in load_state_dict)
    # - decode remains fp-only via global decode policy
    linear_prefill_cfg = {
        "bypass": True,
        "gptq": True,
        "clip_search": False,
        "weight_block_size": 16,
        "weight_exponent_width": 4,
        "weight_frac_width": 3,
        "data_in_block_size": 16,
        "data_in_exponent_width": 4,
        "data_in_frac_width": 3,
        "bias_block_size": 16,
        "bias_exponent_width": 4,
        "bias_frac_width": 3,
    }
    linear_decode_cfg = {
        "bypass": True,
        "weight_block_size": 16,
        "weight_exponent_width": 4,
        "weight_frac_width": 3,
        "data_in_block_size": 16,
        "data_in_exponent_width": 4,
        "data_in_frac_width": 3,
        "bias_block_size": 16,
        "bias_exponent_width": 4,
        "bias_frac_width": 3,
    }

    # MLP/RMS minifloat phase cfg: conservative bypass while preserving schema.
    mlp_rms_phase_cfg = {
        "prefill": {"bypass": True},
        "decode": {"bypass": True},
        "decode_policy": "fp_only",
    }

    return {
        "by": "regex_name",
        "gptq": {
            "model_name": cfg.model_name,
            "device": cfg.device,
            "dataset": cfg.gptq_dataset,
            "nsamples": cfg.gptq_nsamples,
            "seqlen": cfg.gptq_seqlen,
            "format": "mxfp",
            "weight_config": {
                "weight_block_size": 16,
                "weight_exponent_width": 4,
                "weight_frac_width": 3,
            },
            "quantile_search": True,
            "clip_search_y": False,
            "cali_batch_size": cfg.gptq_cali_batch_size,
            "checkpoint_dir": cfg.gptq_checkpoint_dir,
            "hf_token": cfg.hf_token,
            "max_layers": cfg.gptq_max_layers,
        },
        r"model\.layers\.\d+\.self_attn$": {
            "config": {
                "name": "mxfp",
                "decode_policy": "fp_only",
                "prefill": attn_phase_cfg,
                "decode": attn_phase_cfg,
            }
        },
        r"model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)$": {
            "config": {
                "name": "minifloat",
                "decode_policy": "fp_only",
                **mlp_rms_phase_cfg,
            }
        },
        r"model\.layers\.\d+\.mlp$": {
            "config": {
                "name": "mxfp",
                "decode_policy": "fp_only",
                **mlp_rms_phase_cfg,
            }
        },
        r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$": {
            "config": {
                "name": "mxfp",
                "decode_policy": "fp_only",
                "prefill": linear_prefill_cfg,
                "decode": linear_decode_cfg,
            }
        },
    }


def _load_model_and_tokenizer(cfg: ValidationConfig) -> tuple[Any, Any]:
    """Load model/tokenizer from local cache (offline by default)."""

    torch_dtype = _resolve_torch_dtype(cfg.dtype)
    load_kwargs = {
        "local_files_only": cfg.local_files_only,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.hf_token:
        load_kwargs["token"] = cfg.hf_token

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        **load_kwargs,
    )
    model = model.to(cfg.device)
    model.eval()
    return model, tokenizer


def _print_runtime_header(cfg: ValidationConfig) -> None:
    """Print a compact runtime summary for reproducibility."""

    print("=" * 88)
    print("Llama GPTQ + phase-split validator")
    print(f"model_name          : {cfg.model_name}")
    print(f"device              : {cfg.device}")
    print(f"dtype               : {cfg.dtype}")
    print(f"local_files_only    : {cfg.local_files_only}")
    print(f"gptq_dataset        : {cfg.gptq_dataset}")
    print(f"gptq_nsamples       : {cfg.gptq_nsamples}")
    print(f"gptq_seqlen         : {cfg.gptq_seqlen}")
    print(f"gptq_max_layers     : {cfg.gptq_max_layers}")
    print(f"max_new_tokens      : {cfg.max_new_tokens}")
    print("=" * 88)


def _print_cuda_mapping() -> None:
    """Print effective CUDA mapping for traceability.

    Expected setup for this script:
    - launch with CUDA_VISIBLE_DEVICES=1
    - pass --device cuda:0
    """

    if not torch.cuda.is_available():
        print("CUDA not available; validation cannot run on GPU path.")
        return
    idx = torch.cuda.current_device()
    print(f"torch.cuda.current_device(): {idx}")
    print(f"torch.cuda.get_device_name : {torch.cuda.get_device_name(idx)}")


@torch.no_grad()
def _generate_preview(model: Any, tokenizer: Any, cfg: ValidationConfig) -> str:
    """Generate one sample output for human readability inspection."""

    inputs = tokenizer(cfg.prompt, return_tensors="pt").to(cfg.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text


def main() -> None:
    cfg = _parse_args()
    _set_deterministic_seed(cfg.seed)
    _print_runtime_header(cfg)
    _print_cuda_mapping()

    print("[1/4] Loading model and tokenizer from local cache...")
    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer(cfg)
    print(f"      done in {time.time() - t0:.2f}s")

    print("[2/4] Building conservative quantization config...")
    pass_args = _build_conservative_pass_args(cfg)
    print("      done")

    print("[3/4] Running quantize_module_transform_pass (includes GPTQ)...")
    t1 = time.time()
    model, _ = quantize_module_transform_pass(model, pass_args)
    # Replacement may instantiate some wrapper modules on CPU by default.
    # We force a post-pass device sync so generation uses one consistent device.
    model = model.to(cfg.device)
    model.eval()
    print(f"      done in {time.time() - t1:.2f}s")

    print("[4/4] Running generation preview...")
    out = _generate_preview(model, tokenizer, cfg)

    print("\n" + "=" * 88)
    print("Prompt:")
    print(cfg.prompt)
    print("-" * 88)
    print("Generated text preview (manual readability inspection):")
    if cfg.print_chars > 0:
        print(out[: cfg.print_chars])
        if len(out) > cfg.print_chars:
            print(f"\n... [truncated to {cfg.print_chars} chars]")
    else:
        print(out)
    print("=" * 88)


if __name__ == "__main__":
    # Example:
    # CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/validate_llama_quant_gptq_phase_split.py --device cuda:0
    main()
