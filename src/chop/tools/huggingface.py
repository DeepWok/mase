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
    text_column: str = "text",
    tokenizer_checkpoint: str = None,
):
    """
    Tokenizes a dataset using the appropriate tokenizer from Huggingface.

    Args:
    - dataset: str
        Name of the dataset to tokenize.
    - checkpoint: str
        Name of the checkpoint to use for model.
    - return_tokenizer: bool
        Whether to return the tokenizer used for tokenization.
    - return_processor: bool
        Whether to return the processor used for tokenization (for speech models).
    - text_column: str
        Name of the column containing the text to tokenize (default: "text").
    - tokenizer_checkpoint: str
        Optional different checkpoint to use for tokenizer. If None, uses the same as model checkpoint.

    Returns:
    - tokenized_datasets: DatasetDict
        Tokenized dataset.
    - tokenizer: PreTrainedTokenizer
        Tokenizer used for tokenization. Only returned if return_tokenizer is True.
    - processor: PreTrainedProcessor
        Returns processor, which is tokenizer combined with feature extractor, useful for Wav2Vec2 models.
    """
    # Use tokenizer_checkpoint if provided, otherwise use model checkpoint
    tokenizer_ckpt = tokenizer_checkpoint if tokenizer_checkpoint else checkpoint
    logger.info(f"Tokenizing dataset {dataset} with tokenizer from {tokenizer_ckpt} for model {checkpoint}.")

    # Check if this is a wav2vec2 or speech model
    is_speech_model = "wav2vec2" in checkpoint.lower() or (tokenizer_checkpoint and "wav2vec2" in tokenizer_checkpoint.lower())
    
    if is_speech_model:
        logger.info(f"Detected speech model. Using Wav2Vec2Processor for tokenization.")
        processor = Wav2Vec2Processor.from_pretrained(tokenizer_ckpt)
        tokenizer = processor.tokenizer

        # Determine if this is a LibriSpeech dataset
        is_librispeech = "librispeech" in dataset.lower() or "asr" in dataset.lower()
        
        if is_librispeech:
            logger.info(f"Processing LibriSpeech-like ASR dataset: {dataset}")
            
            # Load all available splits
            raw_datasets_dict = {}
            available_splits = ["train.clean.100", "validation.clean", "test.clean"]
            
            # Try specific splits for the condensed_librispeech dataset
            for split in available_splits:
                try:
                    logger.info(f"Loading split: {split}")
                    split_dataset = load_dataset(dataset, split=split, trust_remote_code=True)
                    raw_datasets_dict[split.replace('.', '_')] = split_dataset
                    logger.info(f"Successfully loaded split: {split}")
                except Exception as e:
                    logger.warning(f"Could not load split {split}: {e}")
            
            # If no specific splits could be loaded, try to load the dataset without specifying splits
            if not raw_datasets_dict:
                try:
                    logger.info("Attempting to load dataset without specific split")
                    full_dataset = load_dataset(dataset, trust_remote_code=True)
                    
                    if isinstance(full_dataset, dict) or isinstance(full_dataset, DatasetDict):
                        for split_name, split_dataset in full_dataset.items():
                            raw_datasets_dict[split_name] = split_dataset
                            logger.info(f"Loaded split: {split_name}")
                    else:
                        # If it's a single dataset, just use it as the training set
                        raw_datasets_dict["train"] = full_dataset
                        logger.info("Loaded dataset as single training set")
                except Exception as e:
                    logger.error(f"Failed to load dataset {dataset}: {e}")
                    raise ValueError(f"Could not load dataset {dataset}")
            
            if not raw_datasets_dict:
                logger.error(f"Failed to load any splits for dataset {dataset}")
                raise ValueError(f"Could not load any splits for dataset {dataset}")
            
            # Create a dictionary to hold the tokenized datasets
            tokenized_datasets_dict = {}
            
            # Process each split
            for split_name, split_dataset in raw_datasets_dict.items():
                logger.info(f"Processing split: {split_name}")
                
                # Check which columns are available in the dataset
                available_columns = split_dataset.column_names
                columns_to_remove = []
                for col in ["speaker_id", "file", "id", "chapter_id", "audio"]:
                    if col in available_columns:
                        columns_to_remove.append(col)
                
                # Define preprocessing function for this dataset
                def preprocess_function(example):
                    audio_array = example["audio"]["array"]
                    sampling_rate = example["audio"]["sampling_rate"]

                    inputs = processor.feature_extractor(
                        audio_array, 
                        sampling_rate=sampling_rate
                    )

                    with processor.as_target_processor():
                        labels = processor.tokenizer(example["text"]).input_ids

                    return {
                        "input_values": inputs["input_values"][0],
                        "labels": labels
                    }
                
                # Map the preprocessing function
                logger.info(f"Preprocessing split {split_name}")
                tokenized_dataset = split_dataset.map(
                    preprocess_function,
                    remove_columns=columns_to_remove
                )
                
                # Add to the dictionary with standardized split names
                if "train" in split_name:
                    tokenized_datasets_dict["train"] = tokenized_dataset
                elif "val" in split_name or "validation" in split_name:
                    tokenized_datasets_dict["validation"] = tokenized_dataset
                elif "test" in split_name:
                    tokenized_datasets_dict["test"] = tokenized_dataset
                else:
                    # Keep original name if it doesn't match standard patterns
                    tokenized_datasets_dict[split_name] = tokenized_dataset
            
            # Make sure we have at least train and test splits
            if "train" not in tokenized_datasets_dict:
                # Use the first available split as train
                first_split = list(tokenized_datasets_dict.keys())[0]
                train_data = tokenized_datasets_dict[first_split]
                if "test" in tokenized_datasets_dict:
                    # Already have test, just add train
                    tokenized_datasets_dict["train"] = train_data
                else:
                    # Split the data into train and test
                    train_test_split = train_data.train_test_split(test_size=0.2)
                    tokenized_datasets_dict["train"] = train_test_split["train"]
                    tokenized_datasets_dict["test"] = train_test_split["test"]
            elif "test" not in tokenized_datasets_dict and "validation" not in tokenized_datasets_dict:
                # Need to create a test set
                train_data = tokenized_datasets_dict["train"]
                train_test_split = train_data.train_test_split(test_size=0.2)
                tokenized_datasets_dict["train"] = train_test_split["train"]
                tokenized_datasets_dict["test"] = train_test_split["test"]
            
            # Create the final DatasetDict
            tokenized_datasets = DatasetDict(tokenized_datasets_dict)
            logger.info(f"Created dataset with splits: {list(tokenized_datasets.keys())}")
            
            if return_processor and return_tokenizer:
                return tokenized_datasets, tokenizer, processor
            elif return_tokenizer:
                return tokenized_datasets, tokenizer
            elif return_processor:
                return tokenized_datasets, processor
            else:
                return tokenized_datasets
        else:
            logger.error(f"Dataset type not recognized for speech model: {dataset}")
            raise NotImplementedError(f"Dataset {dataset} not implemented for speech models yet")
    
    # Standard text datasets (non-speech models)
    logger.info(f"Using AutoTokenizer for text dataset tokenization")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    
    # Load and tokenize datasets
    try:
        raw_datasets = load_dataset(dataset)
    except Exception as e:
        logger.warning(f"Failed to load dataset directly: {e}")
        try:
            # Try with streaming
            logger.info("Attempting to load with streaming")
            raw_datasets = load_dataset(dataset, streaming=True)
            # Convert streaming dataset to regular dataset
            dataset_dict = {}
            for split_name, split_dataset in raw_datasets.items():
                dataset_dict[split_name] = Dataset.from_list(list(split_dataset))
            raw_datasets = DatasetDict(dataset_dict)
        except Exception as e2:
            logger.error(f"Failed to load dataset: {e2}")
            raise ValueError(f"Could not load dataset {dataset}")
    
    def tokenize_function(example):
        return tokenizer(
            example[text_column] if text_column in example else example["text"],
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
