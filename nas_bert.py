import os

import numpy as np

from torchvision import transforms
from torchvision.datasets import FakeData

from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers.models.bert.configuration_bert import BertConfig
from datasets import load_dataset
import evaluate

import nni
import nni.nas.strategy as strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment, NasExperimentConfig

from chop.actions.search.search_space import NasBertSpace

# * Config
# * ------------------------------------------

EPOCHS_PER_TRIAL = 1
checkpoint = "bert-base-uncased"
os.environ["WANDB_DISABLED"] = "true"

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

# * Evaluator
# * ------------------------------------------

def fit(model):
    """
    Train the model using HuggingFace trainer, the call onto evaluate library to get the accuracy
    """
    tokenized_datasets, data_collator, tokenizer = get_datasets()
    training_args = TrainingArguments("test-trainer", report_to=None)
    training_args.num_train_epochs = EPOCHS_PER_TRIAL
    training_args.learning_rate = 1e-4

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
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("glue", "sst2")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    nni.report_final_result(results["accuracy"])

# * Define and run experiment
# * ------------------------------------------

cf = BertConfig.from_pretrained(checkpoint)
cf._attn_implementation = "eager"
cf.num_hidden_layers = 1
cf.space_intermediate_size = [768, 1536, 3072, 4096, 6144]
space = NasBertSpace(cf)

model = space.random()

evaluator = FunctionalEvaluator(fit)
strat = strategy.Random()

experiment_config = NasExperimentConfig.default(space, evaluator, strat)
experiment_config.max_trial_number = 3   # spawn 3 trials at most
experiment_config.trial_concurrency = 3  # will run 1 trial concurrently
experiment_config.trial_gpu_number = 3
experiment_config.training_service.use_active_gpu = True

experiment = NasExperiment(space, evaluator, strat, config=experiment_config)
experiment.run(port=8082)

# Sleep forever
import time
while True:
    time.sleep(60)
