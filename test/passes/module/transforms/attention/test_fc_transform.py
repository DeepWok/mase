#!/usr/bin/env python3
import datasets
from datasets import load_dataset as original_load_dataset
import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from chop.tools import get_tokenized_dataset
from chop.passes.module.transforms.attention import fc_transform_pass
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(Path(__file__).resolve().parents[5].as_posix())

def patched_load_dataset(dataset, *args, **kwargs):
    if dataset == "wikitext" and "config" not in kwargs:
        return original_load_dataset(dataset, "wikitext-2-raw-v1", *args, **kwargs)
    else:
        return original_load_dataset(dataset, *args, **kwargs)

datasets.load_dataset = patched_load_dataset

def prepare_dataset():
    logger.info("Loading and preparing dataset...")
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define tokenization function with block size
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    
    # Debug info
    logger.info(f"Test dataset size: {len(tokenized_dataset['test'])}")
    if len(tokenized_dataset['test']) > 0:
        logger.info(f"First test example: {tokenized_dataset['test'][0]}")
    
    logger.info("Filtering dataset...")
    filtered_test = tokenized_dataset["test"].filter(
        lambda x: len(x["input_ids"]) > 0, 
        desc="Filtering empty examples"
    )
    
    logger.info(f"Filtered test dataset size: {len(filtered_test)}")
    if len(filtered_test) > 0:
        logger.info(f"First filtered test example: {filtered_test[0]}")
        logger.info(f"Input IDs length: {len(filtered_test[0]['input_ids'])}")
    
    return filtered_test, tokenizer

def main():
    test_dataset, tokenizer = prepare_dataset()
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No masked language modeling for GPT-2
    )
    
    logger.info("Loading model...")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Apply FC transformation
    logger.info("Applying FC transformation...")
    module_name = "transformer.h.11.attn"
    transformed_model = fc_transform_pass(model, module_name, config={})
    
    training_args = TrainingArguments(
        output_dir="./results_fc_wikitext",
        per_device_eval_batch_size=1,  
        do_eval=True,
        eval_strategy="no",
        report_to="none",
    )
    
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=transformed_model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Try to evaluate first example only for debugging
    logger.info("Testing with a single example...")
    single_example = test_dataset.select([0])
    
    try:
        batch = data_collator([single_example[0]])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        logger.info(f"Batch keys: {batch.keys()}")
        for k, v in batch.items():
            logger.info(f"Batch[{k}].shape: {v.shape}")
        
        with torch.no_grad():
            outputs = transformed_model(**batch)
        logger.info("Single example forward pass successful!")
        
        logger.info("Evaluating full dataset...")
        eval_results = trainer.evaluate()
        eval_loss = eval_results["eval_loss"]
        perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
        
        logger.info("=== WikiText FC Evaluation ===")
        logger.info(f"Eval Loss (Cross Entropy): {eval_loss:.4f}")
        logger.info(f"Perplexity: {perplexity:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())

    with open("fc_transform_results.txt", "w") as f:
        f.write("WikiText FC Evaluation Results:\n")
        f.write(f"Eval Loss (Cross Entropy): {eval_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.4f}\n")

if __name__ == "__main__":
    main()