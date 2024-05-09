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

checkpoint = "bert-base-uncased"
os.environ["WANDB_DISABLED"] = "true"
NUM_TRIALS = 3
DEGREE = 3

# * Utils
# * ------------------------------------------

def get_datasets():
    raw_datasets = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized_datasets, data_collator, tokenizer

def get_model():
    cf = AutoConfig.from_pretrained(checkpoint)
    cf.num_hidden_layers = 3
    model = AutoModelForSequenceClassification.from_config(cf)
    return model

tokenized_datasets, data_collator, tokenizer = get_datasets()
model = get_model()

training_args = TrainingArguments("test-trainer", report_to=None)
training_args.num_train_epochs = 1



# * Search
# * ------------------------------------------

linear_layers = [name for (name, cls) in model.named_modules() if isinstance(cls, torch.nn.Linear)]
print(f"Space size: {2 ** len(linear_layers)}")

cols = linear_layers + ["accuracy", "f1"]
df = pd.DataFrame(columns=cols)

for trial in range(NUM_TRIALS):    
    print(f"\n\n==========================================")
    print(f"Trial: {trial}")
    print(f"==========================================")
    trial_model = get_model()
    linear_layers = [name for (name, cls) in trial_model.named_modules() if isinstance(cls, torch.nn.Linear)]
    layer_included = {layer: random.choice([True, False]) for layer in linear_layers}
    included_layers = [layer for layer, included in layer_included.items() if included]
    print(f"Replacing layers: {included_layers}")
    
    for layer_name in included_layers:
        old_layer = deepgetattr(trial_model, layer_name)
        deepsetattr(trial_model, layer_name, ChebyKANLayer(
            input_dim=old_layer.in_features, output_dim=old_layer.out_features, degree=DEGREE
        ))
        
    # print(f"Pre training weights: {deepgetattr(trial_model, included_layers[0]).cheby_coeffs}")

    trainer = Trainer(
        trial_model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    breakpoint()
    trainer.train()

    # print(f"Post training weights: {deepgetattr(trial_model, included_layers[0]).cheby_coeffs}")
    
    predictions = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("glue", "mrpc")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    df.loc[trial] = {
        **layer_included,
        **results
    }
    
    print(f"Results: {results}")
    
df.to_csv(f"results.csv")

