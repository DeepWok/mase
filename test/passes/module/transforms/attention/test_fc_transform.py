#!/usr/bin/env python3
import datasets
from datasets import load_dataset as original_load_dataset
import math
import torch
import time
import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from chop.tools import get_tokenized_dataset
from chop.passes.module.transforms.attention import fc_transform_pass
from pathlib import Path
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(Path(__file__).resolve().parents[5].as_posix())

def patched_load_dataset(dataset, *args, **kwargs):
    if dataset == "wikitext" and "config" not in kwargs:
        return original_load_dataset(dataset, "wikitext-2-raw-v1", *args, **kwargs)
    else:
        return original_load_dataset(dataset, *args, **kwargs)

datasets.load_dataset = patched_load_dataset

# --------------------------------------------------
# 1. Dataset preparation functions
# --------------------------------------------------
def prepare_dataset():
    logger.info("Loading and preparing dataset...")
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define tokenization function with block size
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    # Tokenize dataset
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing dataset - {split}",
        )
    
    # Additional filtering to remove empty inputs
    filtered_dataset = {}
    for split in tokenized_dataset:
        filtered_dataset[split] = tokenized_dataset[split].filter(
            lambda x: len(x["input_ids"]) > 0, 
            desc=f"Filtering empty examples - {split}"
        )
        logger.info(f"{split} dataset size: {len(filtered_dataset[split])}")
    
    return filtered_dataset, tokenizer

# --------------------------------------------------
# 2. Model evaluation function
# --------------------------------------------------
def evaluate_model(model, test_dataset, tokenizer, model_name="Model"):
    """Evaluate a model and return metrics"""
    logger.info(f"Evaluating {model_name}...")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No masked language modeling for GPT-2
    )
    
    # Set evaluation arguments
    eval_args = TrainingArguments(
        output_dir=f"./results_{model_name.lower().replace(' ', '_')}",
        per_device_eval_batch_size=4,
        do_eval=True,
        eval_strategy="no",
        report_to="none",
        logging_steps=100,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Measure evaluation time
    start_time = time.time()
    eval_results = trainer.evaluate()
    eval_time = time.time() - start_time
    
    # Calculate metrics
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    
    # Print results
    print("\n" + "="*50)
    print(f"{model_name} Evaluation Results:")
    print("="*50)
    print(f"Eval Loss (Cross Entropy): {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    print("="*50)
    
    return {
        "model_name": model_name,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "eval_time": eval_time,
        "raw_results": eval_results
    }

# --------------------------------------------------
# 3. Model training function
# --------------------------------------------------
def train_model(model, train_dataset, eval_dataset, tokenizer, model_name="Model", num_epochs=1):
    """Train a model and return metrics"""
    logger.info(f"Training {model_name}...")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No masked language modeling for GPT-2
    )
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=f"./trained_{model_name.lower().replace(' ', '_')}",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        gradient_accumulation_steps=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Measure training time
    start_time = time.time()
    train_results = trainer.train()
    train_time = time.time() - start_time
    
    # Save model
    trainer.save_model(f"./final_{model_name.lower().replace(' ', '_')}")
    
    # Evaluate after training
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    
    # Print results
    print("\n" + "="*50)
    print(f"{model_name} Training Results:")
    print("="*50)
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Final Eval Loss: {eval_loss:.4f}")
    print(f"Final Perplexity: {perplexity:.4f}")
    print("="*50)
    
    return {
        "model_name": model_name,
        "train_time": train_time,
        "final_eval_loss": eval_loss,
        "final_perplexity": perplexity,
        "train_results": train_results,
        "eval_results": eval_results
    }

# --------------------------------------------------
# 4. Main comparison function
# --------------------------------------------------
def compare_models(do_train=True, num_epochs=1):
    """Compare original and FC-transformed models"""
    # Prepare dataset
    dataset, tokenizer = prepare_dataset()
    
    # 1. Original Model
    logger.info("Loading original model...")
    original_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    original_model.config.pad_token_id = tokenizer.eos_token_id
    
    # 2. FC-Transformed Model
    logger.info("Loading and transforming model...")
    transformed_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    transformed_model.config.pad_token_id = tokenizer.eos_token_id
    
    # Apply FC transformation
    module_name = "transformer.h.11.attn"
    transformed_model = fc_transform_pass(transformed_model, module_name, config={})
    
    # Memory usage analysis
    def get_model_size(model):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # Size in MB
    
    original_size = get_model_size(original_model)
    transformed_size = get_model_size(transformed_model)
    
    print("\n" + "="*50)
    print("Model Size Comparison:")
    print("="*50)
    print(f"Original Model Size: {original_size:.2f} MB")
    print(f"Transformed Model Size: {transformed_size:.2f} MB")
    print(f"Size Reduction: {original_size - transformed_size:.2f} MB ({100 * (original_size - transformed_size) / original_size:.2f}%)")
    print("="*50)
    
    # Evaluation phase
    original_eval = evaluate_model(original_model, dataset["test"], tokenizer, "Original GPT-2")
    transformed_eval = evaluate_model(transformed_model, dataset["test"], tokenizer, "FC-Transformed GPT-2")
    
    results = {
        "model_sizes": {
            "original": original_size,
            "transformed": transformed_size,
            "reduction_percent": 100 * (original_size - transformed_size) / original_size
        },
        "evaluation": {
            "original": original_eval,
            "transformed": transformed_eval
        }
    }
    
    # Training phase (optional)
    if do_train:
        logger.info("Starting training comparison...")
        
        original_train = train_model(original_model, dataset["train"], dataset["validation"], 
                                    tokenizer, "Original GPT-2", num_epochs)
        
        transformed_train = train_model(transformed_model, dataset["train"], dataset["validation"], 
                                       tokenizer, "FC-Transformed GPT-2", num_epochs)
        
        results["training"] = {
            "original": original_train,
            "transformed": transformed_train
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"model_comparison_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print comparison summary
    print("\n" + "="*50)
    print("Model Comparison Summary:")
    print("="*50)
    print(f"Original Model Perplexity: {original_eval['perplexity']:.4f}")
    print(f"FC-Transformed Model Perplexity: {transformed_eval['perplexity']:.4f}")
    print(f"Perplexity Impact: {((transformed_eval['perplexity'] - original_eval['perplexity']) / original_eval['perplexity'] * 100):.2f}%")
    print()
    print(f"Original Model Eval Time: {original_eval['eval_time']:.2f} seconds")
    print(f"FC-Transformed Model Eval Time: {transformed_eval['eval_time']:.2f} seconds")
    print(f"Speed Improvement: {((original_eval['eval_time'] - transformed_eval['eval_time']) / original_eval['eval_time'] * 100):.2f}%")
    print()
    print(f"Size Reduction: {results['model_sizes']['reduction_percent']:.2f}%")
    print("="*50)
    
    if do_train:
        print("\nTraining Results:")
        print(f"Original Model Training Time: {results['training']['original']['train_time']:.2f} seconds")
        print(f"FC-Transformed Model Training Time: {results['training']['transformed']['train_time']:.2f} seconds")
        print(f"Training Speed Improvement: {((results['training']['original']['train_time'] - results['training']['transformed']['train_time']) / results['training']['original']['train_time'] * 100):.2f}%")
        print()
        print(f"Original Model Final Perplexity: {results['training']['original']['final_perplexity']:.4f}")
        print(f"FC-Transformed Model Final Perplexity: {results['training']['transformed']['final_perplexity']:.4f}")
        print("="*50)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare original GPT-2 with FC-transformed version")
    parser.add_argument("--train", action="store_true", help="Include training comparison")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Run comparison
    compare_models(do_train=args.train, num_epochs=args.epochs)