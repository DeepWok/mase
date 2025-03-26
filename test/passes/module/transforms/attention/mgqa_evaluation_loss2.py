import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Example
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from chop.passes.module.transforms import attention_transform_pass

def apply_mgqa_transform(model, kv_heads=2):
    """Apply MGQA transform with a given 'kv_heads' count."""
    pass_args = {
        "by": "type",
        "gpt2spda": { # Assuming 'gpt2spda' is the correct key
            "config": {
                "name": "mgqa",
                "kv_heads": kv_heads,
            }
        },
    }
    transformed_model, _ = attention_transform_pass(model.cpu(), pass_args)
    return transformed_model

def train_gpt2_on_wikitext2(
    model, 
    model_name="gpt2",
    output_dir="./gpt2-finetuned",
    num_train_epochs=1,
    batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    max_seq_length=512,
    save_steps=500,
    eval_steps=500,
    resume_from_checkpoint=True,
):
    """
    Fine-tune a GPT2 model on the WikiText-2 dataset using Hugging Face's Trainer.
    
    Returns:
        tuple: (model, tokenizer, evaluation results)
    """
    # Load the WikiText-2 dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Tokenize the dataset
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Using CLM (Causal Language Modeling) not MLM
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        learning_rate=learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        warmup_steps=500,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    
    # Check if a checkpoint exists
    last_checkpoint = None
    if os.path.isdir(output_dir) and resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            print(f"Resuming from checkpoint: {last_checkpoint}")
    
    # Train model
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
    
    eval_results["perplexity"] = perplexity.item()
    print(f"Evaluation perplexity: {perplexity.item():.2f}")
    
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
    
    return model, tokenizer, eval_results

if __name__ == "__main__":
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    mgqa_model = apply_mgqa_transform(base_model, kv_heads=4)
    output_dir = "./gpt2-mgqa-finetuned"

    model, tokenizer, eval_results = train_gpt2_on_wikitext2(
        mgqa_model,
        output_dir=output_dir,
        batch_size=4,        # Smaller batch size as MGQA could be memory intensive
        num_train_epochs=3,  # Train for more epochs
        learning_rate=3e-5   # Slightly lower learning rate
    )
    model, tokenizer, eval_results = train_gpt2_on_wikitext2(model)