"""
Experiment 2: Training Equivalence — FlexAttention vs SDPA

Verifies that FlexAttention produces identical training dynamics to SDPA.
Trains both for 50 steps on synthetic data with the exact same initial 
weights and compares both per-step loss AND gradient norms.

Model: Tiny Llama (hidden=256, 4 layers, 8 heads, 2 kv_heads)
Dtype: float32 (for precise numerical comparison)

Usage:
    python -u experiments/flex_attention/exp2_training_equivalence.py
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
torch._dynamo.config.cache_size_limit = 64

# ============================================================================
# Config
# ============================================================================

HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
NUM_LAYERS = 4
NUM_HEADS = 8
NUM_KV_HEADS = 2
VOCAB_SIZE = 512
BATCH_SIZE = 4
SEQ_LEN = 256
NUM_STEPS = 50
LR = 1e-3
SEED = 42
DEVICE = "cuda"
DTYPE = torch.float32  # float32 for precise comparison

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# Helpers
# ============================================================================

def make_model(use_flex=False, state_dict=None):
    fat._compiled_flex_attention = None

    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(DTYPE).to(DEVICE)

    # Force strict identical initialization
    if state_dict is not None:
        model.load_state_dict(state_dict)

    if use_flex:
        model, _ = flex_attention_transform_pass(model, {"score_mod": "causal"})

    return model


def train_loop(model, label):
    """Run NUM_STEPS training steps, return per-step losses and grad norms."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    grad_norms = []

    for step in range(NUM_STEPS):
        # Deterministic batch generation
        torch.manual_seed(SEED + step)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Track gradient norm of the first layer's q_proj to prove backprop equivalence
        grad_norm = model.model.layers[0].self_attn.q_proj.weight.grad.norm().item()
        
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        losses.append(loss_val)
        grad_norms.append(grad_norm)
        
        if step % 10 == 0:
            print(f"  [{label}] step {step:3d}: loss = {loss_val:.6f} | grad_norm = {grad_norm:.6f}", flush=True)

    return losses, grad_norms


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Experiment 2: Training Equivalence — FlexAttention vs SDPA")
    print("=" * 70)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name()})")
    print(f"Model: Tiny Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})")
    print(f"Steps: {NUM_STEPS}, Batch: {BATCH_SIZE}, Seq: {SEQ_LEN}")
    print(f"Dtype: {DTYPE}, LR: {LR}, Seed: {SEED}")
    print()

    # --- Generate Master Weights ---
    print("Generating master weights for strict equivalence...")
    torch.manual_seed(SEED)
    base_model = make_model(use_flex=False)
    master_state_dict = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
    del base_model
    torch.cuda.empty_cache()
    print("Done.\n")

    # --- Train SDPA ---
    print("-" * 70)
    print("Training: SDPA causal")
    print("-" * 70)
    model_sdpa = make_model(use_flex=False, state_dict=master_state_dict)
    losses_sdpa, grads_sdpa = train_loop(model_sdpa, "SDPA")
    del model_sdpa
    torch.cuda.empty_cache()
    print()

    # --- Train Flex ---
    print("-" * 70)
    print("Training: Flex causal")
    print("-" * 70)
    model_flex = make_model(use_flex=True, state_dict=master_state_dict)
    losses_flex, grads_flex = train_loop(model_flex, "Flex")
    del model_flex
    torch.cuda.empty_cache()
    print()

    # --- Compare ---
    print("=" * 70)
    print("COMPARISON: Loss & Gradients")
    print("=" * 70)
    print(f"{'Step':>5} | {'SDPA Loss':>10}  {'Flex Loss':>10}  {'Diff':>10} | {'SDPA Grad':>10}  {'Flex Grad':>10}  {'Diff':>10}")
    print("-" * 85)

    loss_diffs = []
    grad_diffs = []
    for i in range(NUM_STEPS):
        l_diff = abs(losses_sdpa[i] - losses_flex[i])
        g_diff = abs(grads_sdpa[i] - grads_flex[i])
        loss_diffs.append(l_diff)
        grad_diffs.append(g_diff)
        if i % 5 == 0:
            print(f"{i:5d} | {losses_sdpa[i]:10.6f}  {losses_flex[i]:10.6f}  {l_diff:10.8f} | {grads_sdpa[i]:10.6f}  {grads_flex[i]:10.6f}  {g_diff:10.8f}")

    max_loss_diff = max(loss_diffs)
    max_grad_diff = max(grad_diffs)
    
    print(f"\nMax Absolute Loss Diff: {max_loss_diff:.8f}")
    print(f"Max Absolute Grad Diff: {max_grad_diff:.8f}")

    if max_loss_diff < 1e-4 and max_grad_diff < 1e-4:
        print("PASS: Training dynamics are mathematically equivalent.")
    else:
        print("WARNING: Significant divergence detected in training dynamics.")

    # --- Save ---
    results = {
        "experiment": "exp2_training_equivalence",
        "config": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "num_steps": NUM_STEPS,
            "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
            "lr": LR, "seed": SEED, "dtype": str(DTYPE),
            "device": torch.cuda.get_device_name(),
        },
        "metrics": {
            "losses_sdpa": losses_sdpa,
            "losses_flex": losses_flex,
            "grads_sdpa": grads_sdpa,
            "grads_flex": grads_flex,
            "max_loss_diff": max_loss_diff,
            "max_grad_diff": max_grad_diff,
        }
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp2_training_equivalence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()