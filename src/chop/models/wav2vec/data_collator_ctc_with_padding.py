import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that dynamically pads the inputs.
    
    Args:
        processor (Wav2Vec2Processor):
            The processor used for processing the data.
        padding (bool or str, optional): Strategy for padding (e.g., True, 'longest', or 'max_length').
        max_length (int, optional): Maximum length for input_values.
        max_length_labels (int, optional): Maximum length for labels.
        pad_to_multiple_of (int, optional): Pad the input_values to a multiple of this value.
        pad_to_multiple_of_labels (int, optional): Pad the labels to a multiple of this value.
    """
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(self, processor: Wav2Vec2Processor, **kwargs):
        self.processor = processor
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, features: List[Union[Dict[str, Any], tuple]]) -> Dict[str, torch.Tensor]:
        # Check the type of the first element to decide how to extract values.
        if isinstance(features[0], dict):
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
        elif isinstance(features[0], (list, tuple)):
            # Assumes the first element is input_values and second is labels.
            input_features = [{"input_values": feature[0]} for feature in features]
            label_features = [{"input_ids": feature[1]} for feature in features]
        else:
            raise TypeError("Each feature must be either a dict or a tuple.")

        # Pad the input features
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Pad the label features in target mode
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # Replace padding in labels with -100 so that they're ignored in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        # Create an attention mask based on the input padding value
        pad_value = self.processor.feature_extractor.padding_value if hasattr(self.processor, "feature_extractor") else 0
        batch["attention_mask"] = (batch["input_values"] != pad_value).long()

        return batch
