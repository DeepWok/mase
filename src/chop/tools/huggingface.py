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
    Wav2Vec2Processor,
)
from datasets import load_dataset, DatasetDict, Dataset
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
    return_processor: bool = False,
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
    - processor: PreTrainedProcessor
        Returns processor, which is tokenizer combined with feature extractor, useful for Wav2Vec2 models.
    """
    logger.info(f"Tokenizing dataset {dataset} with AutoTokenizer for {checkpoint}.")

    if "wav2vec2" in checkpoint:
        if "librispeech_asr" in dataset:
            raw_datasets = load_dataset(dataset, "clean", split="validation", streaming=True, trust_remote_code=True)
            sample_list = list(raw_datasets.take(50))
            small_dataset = Dataset.from_list(sample_list)
        else:
            print("Dataset not implemented yet for wav2vec type models")
            return

        processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
        tokenized_dataset = small_dataset.map(
            lambda x: preprocess_librispeech_asr(x, processor),
            remove_columns=["speaker_id", "file", "id", "chapter_id", "audio"]
        )

        if return_processor:
            return tokenized_dataset, tokenizer, processor
        else:
            return tokenized_dataset, tokenizer
         
    # Load and tokenize datasets
    raw_datasets = load_dataset(dataset)
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
    decoder = None,
    beam_width = 10,
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
            raw_logits = eval_pred.predictions[0]  
            labels = eval_pred.label_ids           

            pred_texts = []

            for i in range(raw_logits.shape[0]):
                sample_logits = torch.from_numpy(raw_logits[i])  
                sample_log_probs = sample_logits.log_softmax(dim=-1).cpu().numpy()

                if decoder is not None:
                    transcription = decoder.decode(sample_log_probs, beam_width=beam_width)
                else:
                    greedy_ids = np.argmax(sample_log_probs, axis=-1)
                    transcription = tokenizer.decode(greedy_ids, skip_special_tokens=True)

                pred_texts.append(transcription.lower())

            # Decode each label individually, filtering out the padding (-100)
            label_texts = []
            for label_seq in labels:
                label_filtered = [token for token in label_seq if token != -100]
                label_text = tokenizer.decode(label_filtered, skip_special_tokens=True)
                label_texts.append(label_text.lower())

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


def preprocess_librispeech_asr(example, processor):
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    inputs = processor(audio=audio_array, sampling_rate=int(sampling_rate), return_tensors="pt", padding=True)
    attention_mask = torch.ones(inputs.input_values.shape, dtype=torch.long)

    with processor.as_target_processor():
        labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids

    return {
        "input_values": inputs.input_values.squeeze(0),
        "attention_mask": attention_mask.squeeze(0),
        "labels": labels.squeeze(0)
    }


