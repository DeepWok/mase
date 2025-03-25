#!/usr/bin/env python3
# Simple script to test MLA transformation with direct module-level fixes

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from chop.passes.module.transforms.attention.mla_fix import apply_all_fixes
from datasets import load_dataset
import os
import logging
import math

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# --- Configuration ---
CHECKPOINT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
OUTPUT_DIR = "./results_mla_test"
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 128
TRAIN_SUBSET_SIZE = 100
INFERENCE_TEXT = "Explain the concept of artificial intelligence in simple terms: "

# Fix for the rotary embedding application in MLA module

def create_dataset(tokenizer):
    """Create a simple dataset for testing"""
    # Load a small subset
    raw_dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=f"train[:{TRAIN_SUBSET_SIZE}]"
    )

    # Simple tokenization with padding
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="np"
        )

    # Process dataset
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Add labels
    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example

    final_dataset = tokenized_dataset.map(add_labels)

    # Split dataset
    split_dataset = final_dataset.train_test_split(test_size=0.1)

    return split_dataset

def finetune_model(model, tokenizer, dataset, output_dir):
    """Run fine-tuning on the model"""
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="no",
        fp16=False,
        bf16=False,
        gradient_accumulation_steps=4,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

    return model

def run_inference(model, tokenizer, text):
    """Run a simple inference test"""
    model.eval()
    device = model.device

    # Prepare input
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate with fallbacks
    with torch.no_grad():
        try:
            # First try with use_cache=True
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        except Exception as e:
            print(f"Generation with cache failed: {e}")
            try:
                # Then try with use_cache=False
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            except Exception as e2:
                print(f"Generation without cache also failed: {e2}")
                # Last resort: direct forward pass
                outputs = model(**inputs).logits.argmax(dim=-1)

    # Decode output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Apply all fixes
    print("\n--- Applying MLA module fixes ---")
    all_fix = apply_all_fixes()


    if not all_fix:
        print("Some fixes could not be applied - proceed with caution")

    # Load model
    print(f"\n--- Loading model from {CHECKPOINT} ---")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # Handle pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move to device
    model.to(device)

    # Save original model
    import copy
    original_model = copy.deepcopy(model)

    # Test original model
    print("\n--- Testing original model ---")
    original_output = run_inference(original_model, tokenizer, INFERENCE_TEXT)
    print(f"Original model output:\n{original_output}")

    # Create dataset
    print("\n--- Creating dataset ---")
    dataset = create_dataset(tokenizer)
    print(f"Dataset created with {len(dataset['train'])} training examples")

    # Fine-tune original model
    print("\n--- Fine-tuning original model ---")
    original_output_dir = f"{OUTPUT_DIR}/original"
    finetune_model(original_model, tokenizer, dataset, original_output_dir)

    # Test fine-tuned original model
    print("\n--- Testing fine-tuned original model ---")
    finetuned_output = run_inference(original_model, tokenizer, INFERENCE_TEXT)
    print(f"Fine-tuned original model output:\n{finetuned_output}")

    # Transform model to MLA
    print("\n--- Transforming model to MLA ---")
    from chop.passes.module.transforms import attention_transform_pass

    # Move to CPU for transformation
    model = model.cpu()

    # Configure transformation
    transform_args = {
        "by": "type",
        "llama": {
            "config": {
                "name": "mla",
                "max_seq_len": MAX_SEQ_LENGTH,
                "max_batch_size": BATCH_SIZE
            }
        },
        "verbose": True
    }

    # Run transformation
    try:
        mla_model, stats = attention_transform_pass(model, transform_args)
        print(f"Transformation stats: {stats}")

        # Verify transformation
        found_mla = False
        for name, module in mla_model.named_modules():
            if 'MLA' in type(module).__name__ or hasattr(module, 'is_mla_wrapper'):
                found_mla = True
                print(f"Found MLA module: {name}")

        if not found_mla:
            print("No MLA modules found after transformation!")

        # Move back to device
        mla_model = mla_model.to(device)
    except Exception as e:
        print(f"Error during transformation: {e}")
        mla_model = original_model
        print("Using original model due to transformation error")

    # Test MLA model
    print("\n--- Testing MLA model (pre-finetune) ---")
    # Disable use_cache if needed
    if hasattr(mla_model.config, "use_cache"):
        orig_use_cache = mla_model.config.use_cache
        mla_model.config.use_cache = False

    mla_output = run_inference(mla_model, tokenizer, INFERENCE_TEXT)
    print(f"MLA model output (pre-finetune):\n{mla_output}")

    # Fine-tune MLA model
    print("\n--- Fine-tuning MLA model ---")
    try:
        mla_output_dir = f"{OUTPUT_DIR}/mla"
        finetune_model(mla_model, tokenizer, dataset, mla_output_dir)
    except Exception as e:
        print(f"Error during MLA fine-tuning: {e}")

    # Test fine-tuned MLA model
    print("\n--- Testing fine-tuned MLA model ---")
    finetuned_mla_output = run_inference(mla_model, tokenizer, INFERENCE_TEXT)
    print(f"Fine-tuned MLA model output:\n{finetuned_mla_output}")

    # Restore use_cache setting
    if hasattr(mla_model.config, "use_cache") and 'orig_use_cache' in locals():
        mla_model.config.use_cache = orig_use_cache

    # Comparison
    print("\n--- Results comparison ---")
    print(f"Original model:\n{original_output}\n")
    print(f"Fine-tuned original model:\n{finetuned_output}\n")
    print(f"MLA model (pre-finetune):\n{mla_output}\n")
    print(f"MLA model (post-finetune):\n{finetuned_mla_output}\n")

if __name__ == "__main__":
    main()