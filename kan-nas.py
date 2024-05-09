# from kan import *
from chop.nn import ChebyKANLayer
import torch
import sys, os, pdb, traceback

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
)
import numpy as np
import evaluate


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


# Set the custom exception hook
sys.excepthook = excepthook

# cf = BertConfig()
# cf.num_hidden_layers = 3

# bert = BertModel(cf)

# bert.encoder.layer[0].attention.self.query = ChebyKANLayer(
#     input_dim=768, output_dim=768, degree=3
# )

# bert(torch.randn((1, 128, 768)))

# * TRAIN
# * ------------------------------------------

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments("test-trainer")


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

model.bert.encoder.layer[0].attention.self.query = ChebyKANLayer(
    input_dim=768, output_dim=768, degree=3
)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)


preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
