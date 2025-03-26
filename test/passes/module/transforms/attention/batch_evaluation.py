#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path
import math
import time # Import time module
import tracemalloc # Import tracemalloc for CPU memory
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
    )

# --- NEW Measurement Function (incorporating attention_mask) ---
def measure_inference_time_and_memory(model, input_ids, attention_mask, device="cpu"):
    """Measures inference time and peak memory for a given model and input batch."""
    try:
        # Ensure device is a torch.device object for consistency
        torch_device = torch.device(device)

        model.to(torch_device)
        model.eval() # Ensure model is in eval mode

        input_ids = input_ids.to(torch_device)
        attention_mask = attention_mask.to(torch_device)

        print(f"  Measuring on device: {torch_device} with input shape: {input_ids.shape}")

        # Warm-up runs
        print("  Warm-up runs...")
        for _ in range(5):
            with torch.no_grad():
                # Pass inputs consistent with model's forward (using keywords recommended)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # --- CUDA Measurement ---
            print("  Measuring CUDA time and memory...")
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats(torch_device)
            initial_memory = torch.cuda.memory_allocated(torch_device)

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            torch.cuda.synchronize()
            end_time = time.time()

            final_memory = torch.cuda.memory_allocated(torch_device)
            peak_memory = torch.cuda.max_memory_allocated(torch_device)
            # Report peak memory relative to initial state (memory used by this inference)
            # Or report absolute peak memory usage on the device
            # Let's report absolute peak for simplicity
            max_memory_mb = peak_memory / (1024 ** 2) # Convert Bytes to MB

            inference_time = end_time - start_time
            print(f"    CUDA Time: {inference_time:.6f} sec")
            print(f"    CUDA Peak Memory: {max_memory_mb:.2f} MB")

        else:
            # --- CPU Measurement ---
            print("  Measuring CPU time and memory...")
            # CPU time is wall time
            # CPU Memory measurement using tracemalloc
            tracemalloc.start()
            start_time = time.time()

            # Record memory before inference
            current_before, peak_before = tracemalloc.get_traced_memory()

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            end_time = time.time()

            # Record memory after inference
            current_after, peak_after = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            inference_time = end_time - start_time
            # Calculate peak memory allocation *during* the timed block
            # Peak_after is the peak during the whole trace; peak_before is peak before start
            # Max memory used *by the operation* is harder to isolate perfectly with tracemalloc
            # We report the peak memory usage during the trace relative to the start
            max_memory_increase_mb = (peak_after - current_before) / (1024 ** 2) # Increase in MB
            absolute_peak_mb = peak_after / (1024 ** 2) # Absolute peak in MB
            max_memory_mb = absolute_peak_mb # Report absolute peak

            print(f"    CPU Time: {inference_time:.6f} sec")
            print(f"    CPU Peak Memory (Absolute): {max_memory_mb:.2f} MB")
            # print(f"    CPU Approx Memory Increase: {max_memory_increase_mb:.2f} MB")

        return inference_time, max_memory_mb

    except Exception as e:
         print(f"Error during measurement: {e}")
         # Ensure tracemalloc is stopped if it was started
         if device != "cuda" and tracemalloc.is_tracing():
             tracemalloc.stop()
         raise e # Re-raise after cleanup attempt

# --- Fine-tuning function (Remains mostly unchanged) ---
def fine_tune_model(model, tokenizer, train_dataset, device_str, args):
    """Fine-tune the model on the provided training dataset (manual loop)."""
    set_reproducible(args.seed)
    device = torch.device(device_str)
    model = model.to(device)
    model.train()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    num_training_steps = len(train_loader) * args.num_train_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    print(f"\n--- Starting Fine-tuning (Manual Loop on {device}) ---")
    print(f"  Epochs: {args.num_train_epochs}, Steps: {num_training_steps}")
    global_step = 0
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    train_start_time = time.time()
    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        epoch_pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(epoch_pbar):
            model.train()
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch.get('attention_mask').to(device)
            labels = batch.get('label').to(device)
            if labels is None: continue
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            if loss is None: continue
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
    train_end_time = time.time()
    print("Fine-tuning finished.")
    print(f"Total Fine-tuning Wall Time: {train_end_time - train_start_time:.2f} sec")
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model

# --- MLA Transformation Function (Using corrected key) ---
def transform_to_mla(model):
    """Apply MLA transform using chop."""
    # Using GPT2Attention as the target key based on previous findings
    pass_args = {
        "by": "type",
        "GPT2Attention": { # Target the standard GPT-2 Attention
            "config": {
                "name": "mla",
                # Add any specific MLA args needed by your 'chop' pass here
            }
        },
    }
    print(f"Applying MLA transform with args: {pass_args}")
    transform_start_time = time.time()
    model_cpu = model.cpu()
    gc.collect()
    try:
        mla_model, _ = attention_transform_pass(model_cpu, pass_args)
        transform_end_time = time.time()
        print(f"MLA transformation complete (took {transform_end_time - transform_start_time:.2f} sec).")
        return mla_model.cpu()
    except Exception as e:
        print(f"Error during attention_transform_pass: {e}")
        if "is not supported" in str(e):
            print("\nHint: The target type 'GPT2Attention' might still be incorrect for your chop version or model.")
            print("      Inspect the model attention class: print(type(model.transformer.h[0].attn))")
        raise e

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure Speed/Memory for GPT-2 (Seq Class) with MLA')
    # Args remain the same as previous script...
    parser.add_argument('--base_model_checkpoint', type=str, default='openai-community/gpt2', help='Base model checkpoint')
    parser.add_argument('--dataset_name', type=str, default='imdb', help='Dataset name')
    parser.add_argument('--finetuned_model_path', type=str, default='./gpt2_finetuned_imdb', help='Path to save/load initial fine-tuned model')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--initial_ft_epochs', type=float, default=0.1, help='Epochs for initial fine-tuning')
    parser.add_argument('--initial_ft_batch_size', type=int, default=8, help='Batch size for initial fine-tuning')
    parser.add_argument('--do_mla_finetune', action='store_true',default=False, help='Perform fine-tuning after MLA transform') # Default False now
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of *MLA* training epochs')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for *MLA* training')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for *MLA* fine-tuning')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for *MLA* fine-tuning')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for preparing the sample input batch') # Used to get sample batch
    # parser.add_argument('--max_train_samples', type=int, default=None, help='Limit train samples') # Keep if needed
    # parser.add_argument('--max_eval_samples', type=int, default=None, help='Limit eval samples') # Keep if needed
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Using device string: {args.device}")
    set_reproducible(args.seed)

    # --- Load Tokenizer ---
    print(f"\nLoading tokenizer: {args.base_model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_checkpoint)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # --- Load or Perform Initial Fine-tuning ---
    # (This section remains the same as before to get the base model state)
    print(f"\nLoading or fine-tuning base model for Sequence Classification...")
    if os.path.exists(args.finetuned_model_path):
        print(f"Loading fine-tuned model from {args.finetuned_model_path}")
        model_config_path = args.finetuned_model_path
        model = AutoModelForSequenceClassification.from_pretrained(
            args.finetuned_model_path, num_labels=2
        )
        if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.eos_token_id
    else:
        print(f"Fine-tuned model not found. Performing initial fine-tuning...")
        model_config_path = args.base_model_checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_checkpoint, num_labels=2
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        raw_dataset_initial = load_dataset(args.dataset_name)
        print("Tokenizing dataset for initial fine-tuning...")
        tokenized_dataset_initial = raw_dataset_initial.map(
             lambda exs: tokenize_function(exs, tokenizer, args.max_length), batched=True, remove_columns=["text"]
        )
        train_dataset_initial = tokenized_dataset_initial["train"]
        eval_dataset_initial = tokenized_dataset_initial["test"]
        training_args_initial = TrainingArguments(
            output_dir=args.finetuned_model_path, num_train_epochs=args.initial_ft_epochs,
            per_device_train_batch_size=args.initial_ft_batch_size, per_device_eval_batch_size=args.initial_ft_batch_size,
            evaluation_strategy="epoch", save_strategy="epoch", logging_steps=100,
            load_best_model_at_end=True, save_total_limit=1, report_to="none"
        )
        trainer_initial = Trainer(model=model, args=training_args_initial, train_dataset=train_dataset_initial,
                                  eval_dataset=eval_dataset_initial, tokenizer=tokenizer)
        print("Starting initial fine-tuning using Hugging Face Trainer...")
        trainer_initial.train()
        print("Initial fine-tuning complete. Saving model.")
        trainer_initial.save_model(args.finetuned_model_path)
        tokenizer.save_pretrained(args.finetuned_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.finetuned_model_path, num_labels=2)
        del trainer_initial, tokenized_dataset_initial, raw_dataset_initial; gc.collect(); torch.cuda.empty_cache()

    base_model_state_dict = copy.deepcopy(model.cpu().state_dict())
    loaded_model_path = args.finetuned_model_path if os.path.exists(args.finetuned_model_path) else args.base_model_checkpoint
    del model; gc.collect(); torch.cuda.empty_cache()
    print("Initial fine-tuned model loaded/created and state stored.")

    # --- Prepare Sample Input Batch for Measurement ---
    print("\nLoading evaluation data to get a sample batch...")
    # Need eval dataset only to get one batch for consistent measurement
    raw_eval_dataset = load_dataset(args.dataset_name, split='test')
    # Tokenize just enough to get one batch
    # Limit samples for tokenization if dataset is huge
    sample_source_dataset = Dataset.from_dict(raw_eval_dataset[:args.eval_batch_size]) # Take first N samples
    print(f"Tokenizing {args.eval_batch_size} samples for measurement input...")
    tokenized_sample_dataset = sample_source_dataset.map(
        lambda exs: tokenize_function(exs, tokenizer, args.max_length),
        batched=True, remove_columns=["text"]
    )
    tokenized_sample_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"]) # Include label though not used in measurement func

    # Create DataLoader and get the sample batch
    sample_loader = torch.utils.data.DataLoader(tokenized_sample_dataset, batch_size=args.eval_batch_size)
    sample_batch = next(iter(sample_loader))
    sample_input_ids = sample_batch['input_ids']
    sample_attn_mask = sample_batch['attention_mask']
    print(f"Sample batch created with input shape: {sample_input_ids.shape}")

    # Clean up unused datasets if memory is tight
    del raw_eval_dataset, sample_source_dataset, tokenized_sample_dataset, sample_loader, sample_batch
    gc.collect()

    # --- Workflow ---
    results = {}

    # Helper to get a fresh model instance from the stored state dict
    def get_fresh_model_from_state(model_path, state_dict):
        # Ensure num_labels=2 for consistency
        model_instance = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        model_instance.load_state_dict(copy.deepcopy(state_dict))
        if model_instance.config.pad_token_id is None:
             model_instance.config.pad_token_id = tokenizer.eos_token_id
        return model_instance

    # 1. Measure Original Fine-tuned Model
    print("\n--- Measuring Original Fine-tuned Model ---")
    try:
        original_model = get_fresh_model_from_state(loaded_model_path, base_model_state_dict)
        # Use the new measurement function
        inf_time, peak_mem = measure_inference_time_and_memory(
            model=original_model,
            input_ids=sample_input_ids,
            attention_mask=sample_attn_mask,
            device=args.device
        )
        print(f"[ORIGINAL FINE-TUNED] Inference Time={inf_time:.6f} sec, Peak Memory={peak_mem:.2f} MB")
        results["original"] = {"time": inf_time, "memory_mb": peak_mem}
        del original_model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"Measurement failed for original model: {e}")
        if 'original_model' in locals(): del original_model
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. Transform and Measure (No Fine-tuning)
    mla_model_no_ft = None
    print("\n--- Transforming to MLA and Measuring (No Fine-tuning) ---")
    try:
        model_to_transform = get_fresh_model_from_state(loaded_model_path, base_model_state_dict).cpu()
        gc.collect()
        mla_model_no_ft = transform_to_mla(model_to_transform) # Returns model on CPU
        del model_to_transform; gc.collect()

        # Use the new measurement function
        inf_time_mla, peak_mem_mla = measure_inference_time_and_memory(
            model=mla_model_no_ft, # Measurement function handles .to(device)
            input_ids=sample_input_ids,
            attention_mask=sample_attn_mask,
            device=args.device
        )
        print(f"[AFTER MLA TRANSFORMATION, NO FINE-TUNING] Inference Time={inf_time_mla:.6f} sec, Peak Memory={peak_mem_mla:.2f} MB")
        results["mla_no_ft"] = {"time": inf_time_mla, "memory_mb": peak_mem_mla}

        if not args.do_mla_finetune:
             del mla_model_no_ft; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
             mla_model_no_ft = None
    except Exception as e:
        print(f"MLA transformation or measurement failed: {e}")
        if mla_model_no_ft is not None: del mla_model_no_ft
        if 'model_to_transform' in locals(): del model_to_transform
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mla_model_no_ft = None

    # 3. Fine-tune MLA Model and Measure (Optional)
    # Note: Need train_dataset loaded for fine-tuning step
    # Load train_dataset here if --do_mla_finetune is true and train_dataset wasn't loaded earlier
    if args.do_mla_finetune:
        print("\n--- Fine-tuning MLA Model and Measuring ---")
        # Load training data only if needed now
        print("Loading and tokenizing training dataset for MLA fine-tuning...")
        raw_train_dataset = load_dataset(args.dataset_name, split='train')
        # Add sample limiting if needed:
        # if args.max_train_samples: raw_train_dataset = raw_train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = raw_train_dataset.map(
            lambda exs: tokenize_function(exs, tokenizer, args.max_length),
            batched=True, remove_columns=["text"]
        )
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        print(f"MLA Training dataset size: {len(train_dataset)}")
        del raw_train_dataset # Free memory

        mla_model_to_ft = None; mla_model_ft = None
        try:
            if mla_model_no_ft is None:
                 print("Recreating MLA model for fine-tuning...")
                 model_to_transform = get_fresh_model_from_state(loaded_model_path, base_model_state_dict).cpu()
                 gc.collect()
                 mla_model_to_ft = transform_to_mla(model_to_transform)
                 del model_to_transform; gc.collect()
            else:
                 print("Using MLA model from previous step for fine-tuning.")
                 mla_model_to_ft = mla_model_no_ft # Should be on CPU

            # Fine-tune
            mla_model_ft = fine_tune_model(
                model=mla_model_to_ft, tokenizer=tokenizer, train_dataset=train_dataset,
                device_str=args.device, args=args
            )
            del train_dataset # Free memory

            # Measure the fine-tuned model
            inf_time_ft, peak_mem_ft = measure_inference_time_and_memory(
                model=mla_model_ft, # Should be on device from fine_tune_model
                input_ids=sample_input_ids,
                attention_mask=sample_attn_mask,
                device=args.device
            )
            print(f"[AFTER MLA FINE-TUNING] Inference Time={inf_time_ft:.6f} sec, Peak Memory={peak_mem_ft:.2f} MB")
            results["mla_ft"] = {"time": inf_time_ft, "memory_mb": peak_mem_ft}

            del mla_model_ft; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            mla_model_ft = None

            if mla_model_to_ft is not None: del mla_model_to_ft; gc.collect(); mla_model_to_ft = None

        except Exception as e:
            print(f"MLA fine-tuning or measurement failed: {e}")
            if mla_model_ft is not None: del mla_model_ft
            if mla_model_to_ft is not None: del mla_model_to_ft
            if mla_model_no_ft is not None: del mla_model_no_ft
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final cleanup
    if not args.do_mla_finetune and mla_model_no_ft is not None:
        del mla_model_no_ft; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Summary ---
    print("\n--- Summary ---")
    def print_speed_mem_result(label, result_dict):
        print(f"{label}: "
              f"Time={result_dict.get('time', float('nan')):.6f} sec, "
              f"Memory={result_dict.get('memory_mb', float('nan')):.2f} MB")

    if "original" in results: print_speed_mem_result("Original Fine-tuned  ", results["original"])
    if "mla_no_ft" in results: print_speed_mem_result("MLA Transformed (No FT)", results["mla_no_ft"])
    if "mla_ft" in results: print_speed_mem_result("MLA Transformed & FT  ", results["mla_ft"])

    print("\nExperiment finished.")