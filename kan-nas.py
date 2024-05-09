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

from transformers import AutoConfig
import random
from copy import deepcopy
import pandas as pd

from chop.nn import ChebyKANLayer
from chop.tools.utils import deepgetattr, deepsetattr


# * Config
# * ------------------------------------------

checkpoint = "prajjwal1/bert-tiny"
os.environ["WANDB_DISABLED"] = "true"
NUM_TRIALS = 100
EPOCHS_PER_TRIAL = 1
DEGREE = 3

# * Utils
# * ------------------------------------------

def get_datasets():
    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized_datasets, data_collator, tokenizer

def get_model():
    cf = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_config(cf)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Num labels: {model.num_labels}")
    return model

tokenized_datasets, data_collator, tokenizer = get_datasets()
model = get_model()

training_args = TrainingArguments("test-trainer", report_to=None)
training_args.num_train_epochs = EPOCHS_PER_TRIAL
training_args.learning_rate = 1e-4


# * Search
# * ------------------------------------------

linear_layers = [name for (name, cls) in model.named_modules() if isinstance(cls, torch.nn.Linear)]
print(f"Space size: {2 ** len(linear_layers)}")

print(f"LAYERS")
for l in linear_layers:
    print(l)
    

cols = linear_layers + ["accuracy", "f1"]
df = pd.DataFrame(columns=cols)

evaluated_configs = []

for trial in range(NUM_TRIALS):    
    print(f"\n\n==========================================")
    print(f"Trial: {trial}")
    print(f"==========================================")
    trial_model = deepcopy(model)
    linear_layers = [name for (name, cls) in trial_model.named_modules() if isinstance(cls, torch.nn.Linear)]
    layer_included = {layer: random.choice([True, False]) for layer in linear_layers}
    included_layers = [layer for layer, included in layer_included.items() if included]
    
    while included_layers in evaluated_configs:
        layer_included = {layer: random.choice([True, False]) for layer in linear_layers}
        included_layers = [layer for layer, included in layer_included.items() if included]
    
    evaluated_configs.append(included_layers)
    
    print(f"Replacing layers: {included_layers}")
    
    for layer_name in included_layers:
        old_layer = deepgetattr(trial_model, layer_name)
        deepsetattr(trial_model, layer_name, ChebyKANLayer(
            input_dim=old_layer.in_features, output_dim=old_layer.out_features, degree=DEGREE
        ))
        
    trainer = Trainer(
        trial_model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    try:
        trainer.train()

        predictions = trainer.predict(tokenized_datasets["validation"])
        preds = np.argmax(predictions.predictions, axis=-1)
        metric = evaluate.load("glue", "sst2")
        results = metric.compute(predictions=preds, references=predictions.label_ids)
        
        df.loc[trial] = {
            **layer_included,
            **results
        }
    
        print(f"Results: {results}")
    except:
        print(f"Trial {trial} failed")
        continue
    
df.to_csv(f"results.csv")

