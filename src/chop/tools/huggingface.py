import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
)
from datasets import load_dataset, DatasetDict
import evaluate

from chop.tools import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")


def get_hf_dummy_in(model):
    """
    Returns a dummy input for a given Huggingface model.

    Args:
    - model: PreTrainedModel
        Huggingface model to get dummy input for.

    Returns:
    - dummy_input: dict
        Dummy input for the model.
    """

    checkpoint = model.config._name_or_path

    logger.info(f"Getting dummy input for {checkpoint}.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    dummy_input = tokenizer(
        [
            "AI may take over the world one day",
            "This is why you should learn ADLS",
        ],
        return_tensors="pt",
    )

    if dummy_input.get("labels", None) is None:
        dummy_input["labels"] = torch.tensor([1, 0])

    return dummy_input


def get_tokenized_dataset(
    dataset: str,
    checkpoint: str,
    return_tokenizer: bool = False,
):
    """
    Tokenizes a dataset using the AutoTokenizer from Huggingface.

    Args:
    - dataset: str
        Name of the dataset to tokenize.
    - checkpoint: str
        Name of the checkpoint to use for tokenization.
    - return_tokenizer: bool
        Whether to return the tokenizer used for tokenization.

    Returns:
    - tokenized_datasets: DatasetDict
        Tokenized dataset.
    - tokenizer: PreTrainedTokenizer
        Tokenizer used for tokenization. Only returned if return_tokenizer is True.
    """
    logger.info(f"Tokenizing dataset {dataset} with AutoTokenizer for {checkpoint}.")

    # Load and tokenize datasets
    raw_datasets = load_dataset(dataset,
        data_files={
            "train": [
                "train-clean-100/61/70968/61-70968-0000.flac",
                "train-clean-100/61/70968/61-70968-0001.flac"
            ],
            "test": [
                "test-clean/61/70968/61-70968-0020.flac"
            ]
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
        )

    # Tokenize
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    if return_tokenizer:
        return tokenized_datasets, tokenizer
    else:
        return tokenized_datasets


def get_trainer(
    model: PreTrainedModel,
    tokenized_dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    evaluate_metric: str = "accuracy",
    data_collator = None,
    output_dir: str = "mase-trainer",
    use_mps_device: bool = False,
    report_to: str = "none",
    num_train_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
):
    """
    Returns a Trainer object for a given model and tokenized dataset.

    Args:
    - model: PreTrainedModel
        Model to train.
    - tokenized_dataset: DatasetDict
        Tokenized dataset.
    - tokenizer: PreTrainedTokenizer
        Tokenizer used for tokenization.
    - evaluate_metric: str
        Metric to use for evaluation.
    - output_dir: str
        Output directory for Trainer.
    - use_cpu: bool
        Whether to use CPU for training.
    - use_mps_device: bool
        Whether to use MPS device for training.
    - report_to: str
        Where to report training results.
    - num_train_epochs: int
        Number of training epochs.

    Returns:
    - trainer: Trainer
        Trainer object for training
    """

    # Handle requested metric
    metric = evaluate.load(evaluate_metric)
    if evaluate_metric == "accuracy":

        def compute_accuracy(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        metric_fn = compute_accuracy

    elif evaluate_metric == "wer":
        def compute_wer(eval_pred):
            logits = eval_pred.predictions[0]
            labels = eval_pred.label_ids
            predictions = np.argmax(logits, axis=-1)

            # Decode each prediction individually
            pred_texts = [tokenizer.decode(pred.tolist(), skip_special_tokens=True) for pred in predictions]
            # Decode each label individually, filtering out the padding (-100)
            label_texts = [
                tokenizer.decode([token for token in label if token != -100], skip_special_tokens=True)
                for label in labels
            ]

            return {"wer": metric.compute(predictions=pred_texts, references=label_texts)}


        metric_fn = compute_wer

    else:
        raise NotImplementedError(f"Metric {metric} not implemented.")

    # Define Trainer
    training_args = TrainingArguments(
        output_dir,
        use_mps_device=use_mps_device,
        report_to=report_to,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        # remove_unused_columns=False, 
    )

    if data_collator is None:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_fn,
    )

    return trainer
