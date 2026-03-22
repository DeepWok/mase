"""
Mini experiment: Train tiny Llama with SDPA (baseline) vs FlexAttention for 3 epochs.

Compares:
  - Per-step training loss curves (should be nearly identical)
  - Per-epoch timing (shows torch.compile warmup amortising after epoch 1)
  - Final loss after training
  - Whether the model actually learns (loss decreases)

Usage:
    python experiments/flex_attention/train_sdpa_vs_flex.py

Output:
    - Printed loss curves
    - JSON results saved to experiments/flex_attention/results/training_comparison.json
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chop.passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)

# ============================================================================
# Config
# ============================================================================
SEED = 42
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
VOCAB_SIZE = 512
MAX_SEQ_LEN = 128
BATCH_SIZE = 8
LR = 1e-3
NUM_EPOCHS = 3
NUM_SAMPLES = 1000  # synthetic dataset size
LOG_EVERY = 10  # print loss every N steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# Synthetic language modelling dataset
# ============================================================================
class SyntheticLMDataset(Dataset):
    """Random token sequences for language modelling.

    Using synthetic data avoids dataset download issues on HPC and keeps
    the experiment self-contained.  Both models see identical data (same seed).
    """

    def __init__(self, num_samples, seq_len, vocab_size, seed=42):
        gen = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            0, vocab_size, (num_samples, seq_len), generator=gen
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


# ============================================================================
# Model factory
# ============================================================================
def make_tiny_llama(seed):
    """Create a tiny Llama with deterministic init."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).to(DEVICE).to(DTYPE)
    return model


# ============================================================================
# Training loop
# ============================================================================
def train_multi_epoch(model, dataloader, lr, num_epochs, tag="model"):
    """Train for multiple epochs, return per-step losses and per-epoch times."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    all_losses = []
    epoch_times = []
    global_step = 0

    for epoch in range(num_epochs):
        epoch_losses = []
        t0 = time.time()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            output = model(input_ids, labels=labels)
            loss = output.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            epoch_losses.append(loss_val)

            if global_step % LOG_EVERY == 0:
                print(f"  [{tag}] epoch {epoch} step {step:4d} | loss = {loss_val:.4f}")
            global_step += 1

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        all_losses.extend(epoch_losses)
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  [{tag}] epoch {epoch} done in {epoch_time:.1f}s | "
              f"avg loss = {avg_loss:.4f}")

    return all_losses, epoch_times


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("FlexAttention vs SDPA Training Comparison")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Model: tiny Llama (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, "
          f"heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS})")
    print(f"Seq len: {MAX_SEQ_LEN}, Batch size: {BATCH_SIZE}, "
          f"Samples: {NUM_SAMPLES}, LR: {LR}, Epochs: {NUM_EPOCHS}")
    print()

    # --- Dataset (identical for both) ---
    dataset = SyntheticLMDataset(NUM_SAMPLES, MAX_SEQ_LEN, VOCAB_SIZE, seed=SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    total_steps = len(dataloader)
    print(f"Total steps per epoch: {total_steps}")
    print()

    # --- SDPA baseline ---
    print("-" * 60)
    print(f"Training SDPA baseline ({NUM_EPOCHS} epochs)...")
    print("-" * 60)
    sdpa_model = make_tiny_llama(SEED)

    sdpa_losses, sdpa_epoch_times = train_multi_epoch(
        sdpa_model, dataloader, LR, NUM_EPOCHS, tag="SDPA"
    )
    sdpa_total = sum(sdpa_epoch_times)
    print(f"  SDPA total: {sdpa_total:.1f}s | final loss = {sdpa_losses[-1]:.4f}")
    print()

    # Free memory before FlexAttention (torch.compile uses extra)
    del sdpa_model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Reset compiled flex_attention cache so it recompiles fresh
    import chop.passes.module.transforms.attention.flex_attention_transform as fat
    fat._compiled_flex_attention = None

    # --- FlexAttention ---
    print("-" * 60)
    print(f"Training FlexAttention (causal score_mod, {NUM_EPOCHS} epochs)...")
    print("-" * 60)
    flex_model = make_tiny_llama(SEED)
    pass_args = {"score_mod": "causal"}
    flex_model, stats = flex_attention_transform_pass(flex_model, pass_args)
    print(f"  Replaced {stats} attention modules")

    flex_losses, flex_epoch_times = train_multi_epoch(
        flex_model, dataloader, LR, NUM_EPOCHS, tag="Flex"
    )
    flex_total = sum(flex_epoch_times)
    print(f"  Flex total: {flex_total:.1f}s | final loss = {flex_losses[-1]:.4f}")
    print()

    # --- Comparison ---
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    sdpa_start = sdpa_losses[0]
    sdpa_end = sdpa_losses[-1]
    flex_start = flex_losses[0]
    flex_end = flex_losses[-1]

    print(f"  SDPA:  loss {sdpa_start:.4f} -> {sdpa_end:.4f}  "
          f"(reduction: {sdpa_start - sdpa_end:.4f})  total time: {sdpa_total:.1f}s")
    print(f"  Flex:  loss {flex_start:.4f} -> {flex_end:.4f}  "
          f"(reduction: {flex_start - flex_end:.4f})  total time: {flex_total:.1f}s")

    # Per-epoch timing comparison
    print(f"\n  Per-epoch timing (seconds):")
    print(f"  {'Epoch':<8} {'SDPA':>10} {'Flex':>10} {'Ratio':>10}")
    print(f"  {'-'*38}")
    for i in range(NUM_EPOCHS):
        ratio = flex_epoch_times[i] / sdpa_epoch_times[i] if sdpa_epoch_times[i] > 0 else float('inf')
        print(f"  {i:<8} {sdpa_epoch_times[i]:>10.1f} {flex_epoch_times[i]:>10.1f} {ratio:>10.2f}x")

    # Check loss curves are similar
    max_diff = max(abs(a - b) for a, b in zip(sdpa_losses, flex_losses))
    mean_diff = sum(abs(a - b) for a, b in zip(sdpa_losses, flex_losses)) / len(sdpa_losses)
    print(f"\n  Loss curve diff: max={max_diff:.6f}, mean={mean_diff:.6f}")

    # Check both models actually learned
    sdpa_learned = sdpa_end < sdpa_start
    flex_learned = flex_end < flex_start
    print(f"  SDPA learned: {sdpa_learned}")
    print(f"  Flex learned: {flex_learned}")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "vocab_size": VOCAB_SIZE,
            "seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
            "num_samples": NUM_SAMPLES,
            "device": DEVICE,
            "dtype": str(DTYPE),
        },
        "sdpa": {
            "losses": sdpa_losses,
            "epoch_times": sdpa_epoch_times,
            "total_time": sdpa_total,
            "start_loss": sdpa_start,
            "final_loss": sdpa_end,
            "learned": sdpa_learned,
        },
        "flex": {
            "losses": flex_losses,
            "epoch_times": flex_epoch_times,
            "total_time": flex_total,
            "start_loss": flex_start,
            "final_loss": flex_end,
            "learned": flex_learned,
        },
        "comparison": {
            "max_loss_diff": max_diff,
            "mean_loss_diff": mean_diff,
            "both_learned": sdpa_learned and flex_learned,
        },
    }

    out_path = RESULTS_DIR / "training_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # --- Pass/fail verdict ---
    print()
    if sdpa_learned and flex_learned and mean_diff < 0.5:
        print("VERDICT: PASS - Both models learn, loss curves are similar")
    elif not flex_learned:
        print("VERDICT: FAIL - FlexAttention model did not learn (loss did not decrease)")
    elif mean_diff >= 0.5:
        print("VERDICT: WARNING - Loss curves diverged significantly")
    else:
        print("VERDICT: FAIL - SDPA baseline did not learn")

    print()


if __name__ == "__main__":
    main()
