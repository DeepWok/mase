"""
Experiment 7: Kernel-Level Profiling

Generates Chrome Tracing / Perfetto profiles to analyze SRAM vs DRAM 
traffic, kernel fusion, and CUDA graph execution differences between 
SDPA and FlexAttention.

Model: Medium Llama, Seq Len: 2048
Usage:
    python -u experiments/flex_attention/exp7_kernel_profiling.py
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
SEQ_LEN = 2048
DEVICE = "cuda"
DTYPE = torch.bfloat16
WINDOW_SIZE = 256

TRACE_DIR = Path(__file__).parent / "results" / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

def make_model(pass_args=None):
    fat._compiled_flex_attention = None
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS, max_position_embeddings=SEQ_LEN,
        vocab_size=VOCAB_SIZE, _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)
    if pass_args is not None:
        model, _ = flex_attention_transform_pass(model, pass_args)
    model.train()
    return model

def create_sliding_window_mask(seq_len, window_size, dtype, device):
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), min_val, dtype=dtype, device=device)
    q_idx, kv_idx = torch.arange(seq_len, device=device).view(-1, 1), torch.arange(seq_len, device=device).view(1, -1)
    mask[0, 0, (q_idx >= kv_idx) & ((q_idx - kv_idx) < window_size)] = 0.0
    return mask.contiguous()

def run_profiler(model, label, attention_mask=None):
    print(f"\nProfiling: {label}...")
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = input_ids.clone()
    fwd_kwargs = {"labels": labels}
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask.expand(BATCH_SIZE, -1, -1, -1)

    # Warmup compiler
    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, **fwd_kwargs)
            outputs.loss.backward()
    torch.cuda.synchronize()

    # Profile exactly 1 step
    trace_name = label.replace(" ", "_").lower()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(TRACE_DIR), worker_name=trace_name)
    ) as prof:
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, **fwd_kwargs)
            outputs.loss.backward()
            torch.cuda.synchronize()
        prof.step()
    
    print(f"  -> Trace saved to {TRACE_DIR}")

def main():
    print("=" * 70)
    print("Experiment 7: Kernel-Level Profiling")
    print("=" * 70)

    # 1. SDPA Causal
    model = make_model()
    run_profiler(model, "SDPA Causal")
    del model; torch.cuda.empty_cache()

    # 2. SDPA SWA
    model = make_model()
    mask = create_sliding_window_mask(SEQ_LEN, WINDOW_SIZE, DTYPE, DEVICE)
    run_profiler(model, "SDPA SWA", attention_mask=mask)
    del model; del mask; torch.cuda.empty_cache()

    # 3. Flex SWA
    model = make_model({"score_mod": "sliding_window", "score_mod_kwargs": {"window_size": WINDOW_SIZE}})
    run_profiler(model, "Flex SWA")
    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()