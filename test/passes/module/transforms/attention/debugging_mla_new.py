#!/usr/bin/env python3
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
import os
from chop.tools import get_trainer
from chop.passes.module.transforms import attention_transform_pass
import torch
torch.cuda.empty_cache()


# 1️⃣ Load dataset
dataset_name = "imdb"
raw_dataset = load_dataset(dataset_name)

# 2️⃣ Load tokenizer
checkpoint = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# 3️⃣ Tokenize dataset
def tokenize_fn(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )

tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# 4️⃣ Check if trained model exists
model_path = "./gpt2_finetuned_imdb"

if os.path.exists(model_path):
    print("[INFO] Loading fine-tuned GPT-2 model from disk...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # accuracy = evaluate.load("accuracy")
    
    # training_args = TrainingArguments(
    #     output_dir=model_path,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     num_train_epochs=0.1,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     logging_steps=100,
    #     load_best_model_at_end=True,
    #     save_total_limit=1,
    # )
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return accuracy.compute(predictions=predictions, references=labels)
    
    # # 7️⃣ Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["test"],
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    # )
    # eval_results_gpt2 = trainer.evaluate()
else:
    print("[INFO] Training GPT-2 model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # 5️⃣ Define evaluation metric (Accuracy)
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    # 6️⃣ Training arguments
    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=100,
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    # 7️⃣ Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Evaluate GPT-2 accuracy
    eval_results_gpt2 = trainer.evaluate()
    print(f"[BEFORE TRANSFORMATION] GPT-2 IMDB Accuracy: {eval_results_gpt2['eval_accuracy']:.4f}")

# 8️⃣ MLA Transformation
def transform_to_mla(model):
    pass_args = {
        "by": "type",
        "gpt2spda": {
            "config": {
                "name": "mla",
            }
        },
    }
    mla_network, _ = attention_transform_pass(model, pass_args)
    print("[INFO] Transformed GPT-2 model to MLA")
    return mla_network

mla_model = transform_to_mla(model)

# 9️⃣ Fine-tune MLA model
trainer_mla = get_trainer(
    model=mla_model,
    tokenized_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=2,
)

print("[INFO] Training MLA-transformed model...")
trainer_mla.train()

# 🔟 Evaluate MLA model
eval_results_mla = trainer_mla.evaluate()
print(f"[AFTER TRANSFORMATION] MLA Evaluation accuracy: {eval_results_mla['eval_accuracy']:.4f}")




