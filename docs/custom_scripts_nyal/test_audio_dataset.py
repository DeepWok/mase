#!/usr/bin/env python
"""
Simple test script to verify that the audio dataset for speech recognition loads correctly.
"""

import os
from pathlib import Path
import torch
from transformers import Wav2Vec2Processor

# Import the dataset from the new audio module
from chop.dataset.audio.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule

def test_audio_dataset_direct():
    """Test loading the dataset directly."""
    print("Testing direct dataset loading...")
    
    # Define a path for dataset storage
    dataset_path = Path("./preprocessed_data")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create the dataset instance
    dataset = CondensedLibrispeechASRDataset(
        path=dataset_path,
        split="train"
    )
    
    # Prepare and setup the dataset
    print("Preparing data (this might take a while if downloading for the first time)...")
    dataset.prepare_data()
    dataset.setup()
    
    # Verify dataset is loaded by checking length and accessing an item
    print(f"Dataset length: {len(dataset)}")
    sample_input, sample_label = dataset[0]
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample label shape: {sample_label.shape}")
    print("Direct dataset loading successful!")
    print("-" * 50)

def test_audio_dataset_via_data_module():
    """Test loading the dataset through MaseDataModule."""
    print("Testing dataset loading through MaseDataModule...")
    
    # Create the data module
    data_module = MaseDataModule(
        name="nyalpatel/condensed_librispeech_asr",
        batch_size=2,
        num_workers=0
    )
    
    # Prepare and setup the data module
    print("Preparing data module...")
    data_module.prepare_data()
    data_module.setup(stage="fit")
    
    # Verify data module contains the dataset correctly
    print("Checking train dataloader...")
    train_dl = data_module.train_dataloader()
    sample_batch = next(iter(train_dl))
    
    # Print information about the loaded batch
    print(f"Batch type: {type(sample_batch)}")
    
    # Handle list return type (as seen in the error)
    if isinstance(sample_batch, list):
        print(f"List length: {len(sample_batch)}")
        for i, item in enumerate(sample_batch):
            if isinstance(item, torch.Tensor):
                print(f"Item {i} shape: {item.shape}")
            else:
                print(f"Item {i} type: {type(item)}")
    # Handle tuple return type
    elif isinstance(sample_batch, tuple):
        inputs, labels = sample_batch
        print(f"Input batch shape: {inputs.shape}")
        print(f"Label batch shape: {labels.shape}")
    # Handle dictionary return type
    elif isinstance(sample_batch, dict):
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"Key: {key}, Shape: {value.shape}")
            else:
                print(f"Key: {key}, Type: {type(value)}")
    else:
        print(f"Unexpected batch type: {type(sample_batch)}")
                
    print("DataModule loading successful!")

if __name__ == "__main__":
    print("=== TESTING AUDIO DATASET FOR SPEECH RECOGNITION ===")
    
    # Test loading the dataset directly
    test_audio_dataset_direct()
    
    # Test loading the dataset through the data module
    test_audio_dataset_via_data_module()
    
    print("All tests completed successfully!")