import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from chop.passes.module.transforms.optical import optical_module_transform_pass

def bert_onn_transform(model):
    pass_args = {
        "by": "type",
        "linear": {
            "config": {
                "name": "morr",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
    }

    name_args = {
        "by": "name",
        "bert.encoder.layer.0.attention.self.query": {
            "config": {
                "name": "morr",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
        "bert.encoder.layer.0.attention.self.key": {
            "config": {
                "name": "morr",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
        "bert.encoder.layer.0.attention.self.value": {
            "config": {
                "name": "morr",
                "miniblock": 4,
                "morr_init": True,
                "trainable_morr_bias": False,
                "trainable_morr_scale": False,
            }
        },
    }
    model, _ = optical_module_transform_pass(model, name_args)
    return model

def test_bert_inference(model, text="This is a test."):
    """
    Passes a sample string through the model for quick debugging.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    return outputs

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print(model)

    # Placeholder for modifications
    model = bert_onn_transform(model)

    dataset = load_dataset("glue", "sst2")
    def preprocess(examples):
        return tokenizer(examples["sentence"], truncation=True, padding=True)
    dataset = dataset.map(preprocess, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return metric.compute(predictions=np.argmax(logits, axis=1), references=labels)

    training_args = TrainingArguments(
        output_dir="model_sst2",
        run_name="bert_sst2_experiment",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    test_bert_inference(model)

    model = bert_onn_transform(model)

    test_bert_inference(model)
    # main()
