#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path
import math
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer, # Still used for initial fine-tuning
    get_linear_schedule_with_warmup,
)
import gc
import numpy as np
from tqdm.auto import tqdm
import argparse
import copy

# Set CUDA device (Consider setting via environment variable externally)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Example

# Chop-specific imports (adjust path as needed)
try:
    # Adjust path based on your project structure
    # sys.path.append(Path(__file__).resolve().parents[2].as_posix())
    from chop.passes.module.transforms import attention_transform_pass
except ImportError:
    print("Warning: 'chop' library not found or path incorrect. MLA transform will fail.")
    def attention_transform_pass(model, pass_args):
        print("Error: 'chop' library needed for attention_transform_pass.")
        raise ImportError("Cannot perform MLA transform without 'chop'.")

# --- Helper Functions ---

def set_reproducible(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Consider commenting out for performance if strict reproducibility isn't paramount
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def tokenize_function(examples, tokenizer, max_length):
    """Tokenization function wrapper for sequence classification."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        # return_tensors="pt" # Let map handle batching, set format later
    )

# ADAPTED from previous script for Sequence Classification
def evaluate_model(model, dataset, batch_size, device, seed=42):
    """Evaluate CE Loss and Perplexity on the dataset (manual loop)."""
    set_reproducible(seed)
    model = model.to(device)
    model.eval()

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    num_batches = 0
    num_samples = 0

    print(f"\n--- Starting Evaluation (Manual Loop) ---")
    print(f"  Eval Batch Size: {batch_size}")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch.get('attention_mask').to(device)
            labels = batch.get('label').to(device) # Use 'label' column for classification

            if labels is None:
                print("Warning: 'label' missing in evaluation batch. Skipping.")
                continue

            batch_size_actual = input_ids.shape[0]

            try:
                outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
                loss = outputs.loss # For sequence classification, loss is typically CE

                if loss is not None:
                    # Accumulate loss, weighted by batch size in case of partial last batch
                    total_loss += loss.item() * batch_size_actual
                    num_batches += 1
                    num_samples += batch_size_actual
                else:
                    print("Warning: Eval loss is None for a batch.")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA out of memory during evaluation with batch size {batch_size}. Try reducing --eval_batch_size.")
                    gc.collect(); torch.cuda.empty_cache()
                    raise e
                else: raise e

    if num_samples == 0:
        print("Warning: No samples processed during evaluation.")
        return float('nan'), float('nan')

    # Calculate average loss over all samples
    avg_loss = total_loss / num_samples
    # Perplexity makes less sense for classification loss, but we calculate it as exp(avg_loss)
    # Often, just the Cross-Entropy loss is reported for classification.
    perplexity = math.exp(avg_loss)

    print("Evaluation finished.")
    gc.collect(); torch.cuda.empty_cache()
    return avg_loss, perplexity

# ADAPTED from previous script for Sequence Classification
def fine_tune_model(model, tokenizer, train_dataset, device, args):
    """Fine-tune the model on the provided training dataset (manual loop)."""
    set_reproducible(args.seed)
    model = model.to(device)
    model.train()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    num_training_steps = len(train_loader) * args.num_train_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"\n--- Starting Fine-tuning (Manual Loop) ---")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Train Batch Size: {args.train_batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Total Steps: {num_training_steps}")
    print(f"  Warmup Steps: {num_warmup_steps}")

    global_step = 0
    total_loss_interval = 0.0
    log_interval = 50 # Log loss every N steps

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        epoch_pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(epoch_pbar):
            model.train()
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch.get('attention_mask').to(device)
            labels = batch.get('label').to(device) # Use 'label' column

            if labels is None:
                 print(f"Warning: 'label' missing in training batch step {global_step}. Skipping.")
                 continue

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss

            if loss is None:
                 print(f"Warning: Training loss is None at step {global_step}. Skipping batch.")
                 continue

            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss_interval += loss.item()
            global_step += 1
            epoch_pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

            if global_step % log_interval == 0:
                avg_loss_interval = total_loss_interval / log_interval
                # print(f"  Step {global_step}/{num_training_steps} - Avg Loss (last {log_interval}): {avg_loss_interval:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
                total_loss_interval = 0.0

    print("Fine-tuning finished.")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model # Return the fine-tuned model

# MLA Transformation Function (from your script)
def transform_to_mla(model):
    """Apply MLA transform using chop."""
    pass_args = {
        "by": "type",
        "gpt2spda": { # Assuming 'gpt2spda' is correct key for chop's GPT2 pass
            "config": {
                "name": "mla",
            }
        },
    }
    print(f"Applying MLA transform...")
    # Apply on CPU to potentially save VRAM during transformation
    mla_model, _ = attention_transform_pass(model.cpu(), pass_args)
    print("MLA transformation complete.")
    return mla_model


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune & Evaluate GPT-2 (Seq Class) with MLA')
    # Model & Data Args
    parser.add_argument('--base_model_checkpoint', type=str, default='openai-community/gpt2', help='Base model checkpoint')
    parser.add_argument('--dataset_name', type=str, default='imdb', help='Dataset name')
    parser.add_argument('--finetuned_model_path', type=str, default='./gpt2_finetuned_imdb', help='Path to save/load initial fine-tuned model')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')

    # Initial Fine-tuning Args (using Trainer)
    parser.add_argument('--initial_ft_epochs', type=float, default=0.1, help='Epochs for initial fine-tuning if model not found (can be float for steps)')
    parser.add_argument('--initial_ft_batch_size', type=int, default=8, help='Batch size for initial fine-tuning')

    # MLA Fine-tuning Args (Manual Loop)
    parser.add_argument('--do_mla_finetune', action='store_true',default=True, help='Perform fine-tuning after MLA transform')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of *MLA* training epochs (Manual Loop)')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for *MLA* training (Manual Loop)')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for *MLA* fine-tuning')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for *MLA* fine-tuning')

    # Evaluation Args
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for evaluation (Manual Loop)')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Limit train samples for MLA fine-tuning')
    parser.add_argument('--max_eval_samples', type=int, default=1000, help='Limit eval samples for quick testing')

    # System Args
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    set_reproducible(args.seed)

    # --- Load Tokenizer ---
    print(f"\nLoading tokenizer: {args.base_model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_checkpoint)
    if tokenizer.pad_token is None:
        print("Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load or Perform Initial Fine-tuning ---
    print(f"\nLoading or fine-tuning base model for Sequence Classification...")
    if os.path.exists(args.finetuned_model_path):
        print(f"Loading fine-tuned model from {args.finetuned_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(args.finetuned_model_path)
        # Ensure pad token id is set correctly after loading
        if model.config.pad_token_id is None:
             model.config.pad_token_id = tokenizer.eos_token_id
    else:
        print(f"Fine-tuned model not found at {args.finetuned_model_path}. Performing initial fine-tuning...")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_checkpoint, num_labels=2 # IMDB has 2 labels
        )
        model.config.pad_token_id = tokenizer.eos_token_id # Set pad token ID

        # Load dataset for initial fine-tuning
        raw_dataset_initial = load_dataset(args.dataset_name)
        print("Tokenizing dataset for initial fine-tuning...")
        tokenized_dataset_initial = raw_dataset_initial.map(
             lambda exs: tokenize_function(exs, tokenizer, args.max_length), batched=True, remove_columns=["text"]
        )
        # Use only a subset for quick initial fine-tuning if needed (or use max_steps in TrainingArguments)
        # train_dataset_initial = tokenized_dataset_initial["train"].select(range(1000))
        # eval_dataset_initial = tokenized_dataset_initial["test"].select(range(200))
        train_dataset_initial = tokenized_dataset_initial["train"]
        eval_dataset_initial = tokenized_dataset_initial["test"]


        training_args_initial = TrainingArguments(
            output_dir=args.finetuned_model_path,
            num_train_epochs=args.initial_ft_epochs,
            per_device_train_batch_size=args.initial_ft_batch_size,
            per_device_eval_batch_size=args.initial_ft_batch_size, # Use same for eval here
            evaluation_strategy="epoch", # Or steps if epochs < 1
            save_strategy="epoch",       # Or steps
            logging_steps=100,
            load_best_model_at_end=True, # Saves the best model
            save_total_limit=1,          # Only keep the best checkpoint
            report_to="none" # Disable wandb/tensorboard reporting for this script
        )
        trainer_initial = Trainer(
            model=model,
            args=training_args_initial,
            train_dataset=train_dataset_initial,
            eval_dataset=eval_dataset_initial, # Evaluate during initial fine-tuning
            tokenizer=tokenizer,
            # compute_metrics can be added here if needed
        )
        print("Starting initial fine-tuning using Hugging Face Trainer...")
        trainer_initial.train()
        print("Initial fine-tuning complete. Saving model.")
        trainer_initial.save_model(args.finetuned_model_path) # Save the best model
        tokenizer.save_pretrained(args.finetuned_model_path)
        # Reload the best model explicitly ensure we have it
        model = AutoModelForSequenceClassification.from_pretrained(args.finetuned_model_path)
        del trainer_initial, tokenized_dataset_initial, raw_dataset_initial
        gc.collect(); torch.cuda.empty_cache()

    # Store the state dict of the fine-tuned base model
    base_model_state_dict = copy.deepcopy(model.state_dict())
    del model # Free memory
    gc.collect(); torch.cuda.empty_cache()
    print("Initial fine-tuned model loaded/created.")


    # --- Load Data for MLA Evaluation/Fine-tuning ---
    print("\nLoading and tokenizing dataset for MLA evaluation/fine-tuning...")
    raw_dataset = load_dataset(args.dataset_name)

    # Training Data (for MLA fine-tuning)
    if args.do_mla_finetune:
        train_data_raw = raw_dataset['train']
        if args.max_train_samples:
            print(f"Limiting MLA training data to {args.max_train_samples} samples.")
            # Shuffle before selecting for more representative subset
            train_data_raw = train_data_raw.shuffle(seed=args.seed).select(range(args.max_train_samples))

        print("Tokenizing MLA training data...")
        train_dataset = train_data_raw.map(
            lambda exs: tokenize_function(exs, tokenizer, args.max_length),
            batched=True, remove_columns=["text"]
        )
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        print(f"MLA Training dataset size: {len(train_dataset)}")
    else:
        train_dataset = None
        print("Skipping MLA training data loading as --do_mla_finetune is not set.")

    # Evaluation Data
    eval_data_raw = raw_dataset['test']
    if args.max_eval_samples:
        print(f"Limiting evaluation data to {args.max_eval_samples} samples.")
        eval_data_raw = eval_data_raw.shuffle(seed=args.seed).select(range(args.max_eval_samples))

    print("Tokenizing evaluation data...")
    eval_dataset = eval_data_raw.map(
        lambda exs: tokenize_function(exs, tokenizer, args.max_length),
        batched=True, remove_columns=["text"]
    )
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    print(f"Evaluation dataset size: {len(eval_dataset)}")


    # --- Workflow ---
    results = {}

    # Helper to get a fresh model instance
    def get_fresh_model(checkpoint_path, state_dict):
        model_instance = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        model_instance.load_state_dict(copy.deepcopy(state_dict))
        # Ensure pad token id is set
        if model_instance.config.pad_token_id is None:
             model_instance.config.pad_token_id = tokenizer.eos_token_id
        return model_instance

    # 1. Evaluate Original Fine-tuned Model
    print("\n--- Evaluating Original Fine-tuned Model (Manual Loop) ---")
    try:
        original_model = get_fresh_model(args.finetuned_model_path, base_model_state_dict)
        ce_before, ppl_before = evaluate_model(
            model=original_model,
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed
        )
        print(f"[ORIGINAL FINE-TUNED] CE={ce_before:.4f}, PPL={ppl_before:.4f}")
        results["original"] = {"loss": ce_before, "ppl": ppl_before}
        del original_model
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"Evaluation failed for original model: {e}")
        if 'original_model' in locals(): del original_model
        gc.collect(); torch.cuda.empty_cache()


    # 2. Transform and Evaluate (No Fine-tuning)
    print("\n--- Transforming to MLA and Evaluating (No Fine-tuning) ---")
    try:
        # Get fresh model instance
        model_to_transform = get_fresh_model(args.finetuned_model_path, base_model_state_dict)
        # Transform
        mla_model_no_ft = transform_to_mla(model_to_transform)
        del model_to_transform; gc.collect() # Clean up pre-transform model

        # Evaluate
        ce_after, ppl_after = evaluate_model(
            model=mla_model_no_ft, # Will be moved to device in evaluate_model
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed
        )
        print(f"[AFTER MLA TRANSFORMATION, NO FINE-TUNING] CE={ce_after:.4f}, PPL={ppl_after:.4f}")
        results["mla_no_ft"] = {"loss": ce_after, "ppl": ppl_after}

        # Keep mla_model_no_ft if not fine-tuning, otherwise clean up
        if not args.do_mla_finetune:
             # We are done with this model if not fine-tuning further
             del mla_model_no_ft; gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        print(f"MLA transformation or initial evaluation failed: {e}")
        if 'mla_model_no_ft' in locals(): del mla_model_no_ft
        if 'model_to_transform' in locals(): del model_to_transform
        gc.collect(); torch.cuda.empty_cache()


    # 3. Fine-tune MLA Model and Evaluate (Optional)
    if args.do_mla_finetune:
        print("\n--- Fine-tuning MLA Model and Evaluating ---")
        try:
            # If previous step succeeded, mla_model_no_ft holds the transformed model (on CPU).
            # If previous step failed, we need to recreate it.
            if 'mla_model_no_ft' not in locals() or mla_model_no_ft is None:
                 print("Recreating MLA model for fine-tuning...")
                 model_to_transform = get_fresh_model(args.finetuned_model_path, base_model_state_dict)
                 mla_model_to_ft = transform_to_mla(model_to_transform)
                 del model_to_transform; gc.collect()
            else:
                 # Use the model from the previous step
                 mla_model_to_ft = mla_model_no_ft # Already exists (on CPU)

            if train_dataset is None:
                 print("Cannot fine-tune MLA model: Training data not loaded (--do_mla_finetune specified without data).")
            else:
                # Fine-tune using manual loop
                mla_model_ft = fine_tune_model(
                    model=mla_model_to_ft, # Will be moved to device in fine_tune_model
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    device=args.device,
                    args=args # Pass training args
                )

                # Evaluate the fine-tuned MLA model
                ce_ft, ppl_ft = evaluate_model(
                    model=mla_model_ft, # Already on device
                    dataset=eval_dataset,
                    batch_size=args.eval_batch_size,
                    device=args.device,
                    seed=args.seed
                )
                print(f"[AFTER MLA FINE-TUNING] CE={ce_ft:.4f}, PPL={ppl_ft:.4f}")
                results["mla_ft"] = {"loss": ce_ft, "ppl": ppl_ft}

                del mla_model_ft # Clean up fine-tuned model
                gc.collect(); torch.cuda.empty_cache()

            # Clean up the potentially unused model from step 2 if fine-tuning happened
            if 'mla_model_no_ft' in locals() and mla_model_no_ft is not None:
                 # Check if it's the same object as mla_model_to_ft (it should be)
                 # If we didn't fine-tune (e.g., no data), mla_model_to_ft might still be mla_model_no_ft
                 if 'mla_model_to_ft' not in locals() or mla_model_no_ft is mla_model_to_ft:
                      del mla_model_no_ft
                      gc.collect(); torch.cuda.empty_cache()
                 # If fine-tuning happened, mla_model_no_ft might be same as mla_model_to_ft before ft
                 # but mla_model_ft is the final result. Ensure intermediate is deleted if distinct.


        except Exception as e:
            print(f"MLA fine-tuning or evaluation failed: {e}")
            if 'mla_model_ft' in locals(): del mla_model_ft
            if 'mla_model_to_ft' in locals(): del mla_model_to_ft
            if 'mla_model_no_ft' in locals(): del mla_model_no_ft
            gc.collect(); torch.cuda.empty_cache()


    # --- Summary ---
    print("\n--- Summary ---")
    if "original" in results:
        print(f"Original Fine-tuned: CE={results['original']['loss']:.4f}, PPL={results['original']['ppl']:.4f}")
    if "mla_no_ft" in results:
        print(f"MLA Transformed (No FT): CE={results['mla_no_ft']['loss']:.4f}, PPL={results['mla_no_ft']['ppl']:.4f}")
    if "mla_ft" in results:
        print(f"MLA Transformed & Fine-tuned: CE={results['mla_ft']['loss']:.4f}, PPL={results['mla_ft']['ppl']:.4f}")

    print("\nExperiment finished.")