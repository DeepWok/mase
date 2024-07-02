import os

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
)
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

NUM_TRIALS = 10
TRIAL_CONCURRENCY = 3
EPOCHS_PER_TRIAL = 5
NUM_LATENCY_EVALUATION_ITERATIONS = 10
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
    training_args.save_strategy = "no"

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # calculate time taken for prediction
    avg_time = 0
    for _ in range(NUM_LATENCY_EVALUATION_ITERATIONS):
        start_time = time.time()
        predictions = trainer.predict(tokenized_datasets["validation"])
        end_time = time.time()
        avg_time += end_time - start_time

    avg_time /= NUM_LATENCY_EVALUATION_ITERATIONS

    # Get the accuracy from the last prediction
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("glue", "sst2")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    nni.report_final_result(
        {
            "accuracy": results["accuracy"],
            "default": results["accuracy"],
            "average_latency": avg_time,
            "average_tps": len(tokenized_datasets["validation"]) / avg_time,
        }
    )


# * Define and run experiment
# * ------------------------------------------

cf = BertConfig.from_pretrained(checkpoint)
cf._attn_implementation = "eager"

# Full model parameters
cf.num_hidden_layers = 3
cf.space_hidden_size = [128, 256, 512, 768, 1024]

# Per layer
cf.space_self_attention_implementation = ["attention", "linear", "feedthrough"]
cf.space_self_attention_layer_norm = ["layer_norm", "identity"]
cf.space_output_layer_norm = ["layer_norm", "identity"]
cf.space_intermediate_size = [192, 384, 768, 1536, 3072]
cf.space_num_attention_heads = [2, 4, 8, 16]

space = NasBertSpace(cf)

evaluator = FunctionalEvaluator(fit)
strat = strategy.TPE()

experiment_config = NasExperimentConfig.default(space, evaluator, strat)
experiment_config.max_trial_number = NUM_TRIALS  # spawn 3 trials at most
experiment_config.trial_concurrency = TRIAL_CONCURRENCY  # will run 1 trial concurrently
experiment_config.trial_gpu_number = 1  # use 1 GPU for each trial
experiment_config.training_service.use_active_gpu = True

experiment = NasExperiment(space, evaluator, strat, config=experiment_config)
experiment.start(port=8081)


def dump_experiment_results(data):
    df = pd.DataFrame(
        columns=["accuracy", "average_latency", "default", "average_tps"], index=[]
    )
    for trial in data:
        df.loc[trial.trialJobId] = trial.value
    df.to_csv(f"experiment_{experiment.id}_results.csv")
    return df


while True:
    if experiment.get_status() == "DONE":
        data = experiment.export_data()
        df = dump_experiment_results(data)
        break

# create random dataframe with 100 trials where x-y follows an exponential distribution

# DEBUG
# np.random.seed(0)
# times = np.linspace(0.5, 100, 100)
# df = pd.DataFrame(
#     {
#         "accuracy": np.log(times) + np.random.normal(0, 0.5, 100),
#         "average_latency": times,
#         "average_tps": times,
#     }
# )


def plot_pareto(df):
    plt.figure()
    plt.scatter(df["average_latency"], df["accuracy"])

    # Plot pareto frontier
    df = df.sort_values("average_latency")
    pareto = df["accuracy"].cummax()
    plt.plot(df["average_latency"], pareto, color="red")

    plt.xlabel("Latency [ms]")
    plt.ylabel("Accuracy [%]")
    plt.title("Accuracy/Latency Pareto Frontier")
    plt.savefig(f"experiment_{experiment.id}_results.png")


# plt a figure with two subplots, one for latency and one for tps
# def plot_pareto(df):
#     fig, ax = plt.subplots(1, 2)
#     ax[0].scatter(df["average_latency"], df["accuracy"])
#     ax[0].set_xlabel("Latency [ms]")
#     ax[0].set_ylabel("Accuracy [%]")
#     ax[0].set_title("Accuracy/Latency Pareto Frontier")

#     ax[1].scatter(df["average_tps"], df["accuracy"])
#     ax[1].set_xlabel("TPS")
#     ax[1].set_ylabel("Accuracy [%]")
#     ax[1].set_title("Accuracy/TPS Pareto Frontier")
#     fig.savefig(f"experiment_{0}_results.png")


plot_pareto(df)
