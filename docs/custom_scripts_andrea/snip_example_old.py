import os
import sys

# Add the src directory to the Python path so that the chop module can be found
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Check for required packages and install them if needed
required_packages = ['dill', 'toml', 'torch', 'torchvision', 'transformers', 'datasets', 'pyctcdecode']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Missing required packages: {', '.join(missing_packages)}")
    print("Please install the missing packages with:")
    print(f"pip install {' '.join(missing_packages)}")
    print("\nAlternatively, you can run this script with PYTHONPATH set to include the src directory:")
    print(f"PYTHONPATH={src_dir} python {__file__}")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import types
import importlib
from torch.utils.data import DataLoader

import logging
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
from pyctcdecode import build_ctcdecoder

from pathlib import Path
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms.pruning.prune import prune_transform_pass
from chop.passes.graph.transforms.pruning.snip_helper import SNIPCallback
import chop.passes as passes
from chop.passes.module import report_trainable_parameters_analysis_pass
from chop.tools import get_tokenized_dataset
from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainerCallback
from chop.dataset.audio.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC
from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from chop.tools import get_trainer
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    onnx_runtime_interface_pass,
    quantize_transform_pass,
)

# Print available pruning methods
import chop.passes.graph.transforms.pruning.prune as prune_mod
importlib.reload(prune_mod)
print("Available pruning methods:", list(prune_mod.weight_criteria_map["local"]["elementwise"].keys()))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------
    # 1. Define the model and dataset
    # -------------------------------
    print("Loading model and dataset...")
    checkpoint = "facebook/wav2vec2-base-960h"
    tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
    dataset_name = "nyalpatel/condensed_librispeech_asr"
    
    # Logic inside get_tockenized_dataset needs to be improved using nyal's changes
    try:
        tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
            dataset=dataset_name,
            checkpoint=checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
            return_tokenizer=True,
            return_processor=True,
        )
        print("Using get_tokenized_dataset for dataset preparation")
    except Exception as e:
        print(f"Error using get_tokenized_dataset: {e}")
        print("Falling back to manual dataset loading")
        
        # Load processor
        processor = Wav2Vec2Processor.from_pretrained(tokenizer_checkpoint)
        tokenizer = processor.tokenizer
        
        # Fall back to direct Hugging Face datasets loading
        dataset = load_dataset(dataset_name, split="validation.clean", trust_remote_code=True)
        sample_list = list(dataset.take(50))
        
        def preprocess_function(example):
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
        
        small_dataset = Dataset.from_list(sample_list)
        small_dataset = small_dataset.map(
            preprocess_function,
            remove_columns=["speaker_id", "file", "id", "chapter_id", "audio"]
        )
        
        tokenized_dataset = DatasetDict({
            "train": small_dataset,
            "test": small_dataset
        })
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
    decoder = build_ctcdecoder(vocab)
    
    # -------------------------------
    # 2. Import dataset
    # -------------------------------
    try:
        # Try the structured dataset approach
        print("Loading dataset using CondensedLibrispeechASRDataset...")
        dataset_path = Path("./preprocessed_data")
        condensed_dataset = CondensedLibrispeechASRDataset(path=dataset_path, split="train")
        condensed_dataset.prepare_data()
        condensed_dataset.setup()
        batch_size = 4
        
        data_module = MaseDataModule(
            name=dataset_name,
            batch_size=batch_size,
            model_name=checkpoint,
            num_workers=0,
        )
        data_module.prepare_data()
        data_module.setup()
        
        print("Successfully loaded dataset with MaseDataModule")
    except Exception as e:
        print(f"Error setting up MaseDataModule: {e}")
        print("Using previously loaded tokenized_dataset")
    
    # Load model
    model = AutoModelForCTC.from_pretrained(checkpoint)
    model.config.gradient_checkpointing = False
    encoder = model.wav2vec2    # static, FX-friendly
    ctc_head = model.lm_head    # dynamic CTC head, separate this
    
    # -------------------------------
    # 3. Define the MASE graph & metadata
    # -------------------------------
    print("Creating MASE graph and computing metadata...")
    mg = MaseGraph(
        encoder,
        hf_input_names=[
            "input_values",
            "attention_mask",
        ],
    )
    mg, _ = passes.init_metadata_analysis_pass(mg)
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": False,
        }
    )
    
    # Create combined model for training
    combined_model = CombinedWav2Vec2CTC(
        encoder=mg.model,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    
    # -------------------------------
    # 4. Print parameter count before pruning
    # -------------------------------
    print("\nParameter count before pruning:")
    try:
        if hasattr(passes.module, "report_trainable_parameters_analysis_pass"):
            _, report_before = passes.module.report_trainable_parameters_analysis_pass(mg.model)
            for key, value in report_before.items():
                print(f"  {key}: {value}")
        else:
            raise AttributeError("report_trainable_parameters_analysis_pass not available")
    except (AttributeError, TypeError) as e:
        print(f"Could not use detailed parameter report: {e}")
        total_params = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
        print(f"  Total trainable parameters: {total_params}")
    
    # -------------------------------
    # 5. Apply SNIP callback for pruning preparation
    # -------------------------------
    print("\nPreparing for SNIP pruning...")
    
    try:
        # Use the trainer approach with SNIPCallback
        trainer = get_trainer(
            model=combined_model,
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            evaluate_metric="wer",
            num_train_epochs=1,
            data_collator=data_collator,
            gradient_accumulation_steps=4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
        )
        
        # Get a representative batch for SNIP computation
        try:
            # Try to get a batch directly from the trainer's dataloader
            print("Getting a batch from trainer's dataloader...")
            train_dataloader = trainer.get_train_dataloader()
            dummy_batch = next(iter(train_dataloader))
        except (AttributeError, StopIteration) as e:
            # Fall back to manually creating a batch with the proper collator
            print(f"Falling back to manual batch creation: {e}")
            # Use the data_collator to properly handle variable-sized inputs
            dummy_batch = data_collator([tokenized_dataset["train"][i] for i in range(min(2, len(tokenized_dataset["train"])))])
        
        # Move tensors to the right device
        for k, v in dummy_batch.items():
            if isinstance(v, torch.Tensor):
                dummy_batch[k] = v.to(device)
        
        # Create the callback with the representative batch
        snip_callback = SNIPCallback(representative_batch=dummy_batch)
        trainer.add_callback(snip_callback)
        
        print("Running warm-up training step to accumulate SNIP gradient data...")
        trainer.train()
        print("Warm-up training complete.")
    except Exception as e:
        print(f"Error using Trainer approach: {e}")
        print("Falling back to direct SNIP computation...")
        
        # Monkey-patch the model modules directly
        original_forwards = {}
        for name, module in mg.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                original_forwards[name] = module.forward  # Save original forward
                
                # Create weight_mask parameter if not present
                if not hasattr(module, "weight_mask"):
                    module.weight_mask = nn.Parameter(torch.ones_like(module.weight))
                
                # Override forward method
                if isinstance(module, nn.Conv2d):
                    def new_forward(self, x):
                        return F.conv2d(
                            x,
                            self.weight.detach() * self.weight_mask,
                            self.bias,
                            self.stride,
                            self.padding,
                            self.dilation,
                            self.groups,
                        )
                    module.forward = types.MethodType(new_forward, module)
                elif isinstance(module, nn.Linear):
                    def new_forward(self, x):
                        return F.linear(x, self.weight.detach() * self.weight_mask, self.bias)
                    module.forward = types.MethodType(new_forward, module)
        
        # Get a dummy input batch
        dummy_batch = next(iter(DataLoader(tokenized_dataset["train"], batch_size=2)))
        
        # Forward-backward pass
        mg.model.zero_grad()
        output = mg.model(
            input_values=dummy_batch["input_values"].to(device),
            attention_mask=dummy_batch["attention_mask"].to(device)
        )
        
        # Use mean of hidden states as a simple loss
        loss = output["last_hidden_state"].abs().mean()
        loss.backward()
        
        # Store gradients
        for name, module in mg.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_mask"):
                grad = module.weight_mask.grad
                if grad is not None:
                    if not hasattr(module, "metadata"):
                        module.metadata = {}
                    if "weight" not in module.metadata:
                        module.metadata["weight"] = {}
                    if "stats" not in module.metadata["weight"]:
                        module.metadata["weight"]["stats"] = {}
                    
                    module.metadata["weight"]["stats"]["snip_scores"] = grad.abs().detach().clone()
                    print(f"Module {name}: SNIP score norm = {grad.abs().norm().item()}")
        
        # Restore original forwards
        for name, module in mg.model.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]
    
    # -------------------------------
    # 6. Apply pruning pass using SNIP method
    # -------------------------------
    print("\nApplying SNIP pruning...")
    for name, module in mg.model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "metadata"):
            stats = module.metadata.get("weight", {}).get("stats", {})
            if "snip_scores" in stats:
                print(f"{name}: snip_scores present, norm = {stats['snip_scores'].norm().item()}")
            else:
                print(f"{name}: snip_scores MISSING")
    
    pruning_config = {
        "weight": {
            "sparsity": 0.5,      # Adjust target sparsity as desired
            "method": "snip",     # Use SNIP ranking function
            "scope": "local",
            "granularity": "elementwise",
        },
        "activation": {
            "sparsity": 0.0,
            "method": "l1-norm",
            "scope": "local",
            "granularity": "elementwise",
        },
    }
    
    mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
    
    # -------------------------------
    # 7. Print parameter count after pruning
    # -------------------------------
    print("\n" + "="*80)
    print("PARAMETER COUNTS AFTER PRUNING")
    print("="*80)
    
    # Get the full model's parameter count (encoder + CTC head)
    full_model_params = sum(p.numel() for p in combined_model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
    ctc_head_params = sum(p.numel() for p in ctc_head.parameters() if p.requires_grad)
    
    print(f"\n1. FULL MODEL (Encoder + CTC Head):")
    print(f"   Total parameters:         {full_model_params:,}")
    print(f"   - Encoder parameters:     {encoder_params:,}")
    print(f"   - CTC Head parameters:    {ctc_head_params:,}")
    
    # Optionally show the report from the analysis pass
    try:
        if hasattr(passes.module, "report_trainable_parameters_analysis_pass"):
            _, report_after = passes.module.report_trainable_parameters_analysis_pass(mg.model)
            if "Total Trainable Parameters" in report_after:
                print(f"\n   Note: Analysis report shows: {report_after['Total Trainable Parameters']:,}")
    except (AttributeError, TypeError) as e:
        pass
    
    # Count remaining parameters (non-pruned) for the encoder only
    print(f"\n2. PRUNING DETAILS (Encoder only - this is what's being pruned):")
    
    remaining_params = 0
    prunable_params_total = 0
    
    for module in mg.model.modules():
        if hasattr(module, 'parametrizations'):
            for param_name, param_list in module.parametrizations.items():
                for param in param_list:
                    if hasattr(param, 'mask'):
                        mask_size = param.mask.numel()
                        prunable_params_total += mask_size
                        remaining_params += param.mask.sum().item()
    
    non_prunable_params = encoder_params - prunable_params_total
    
    print(f"   Total encoder parameters:     {encoder_params:,}")
    print(f"   - Prunable parameters:        {prunable_params_total:,} ({prunable_params_total/encoder_params:.1%} of encoder)")
    print(f"   - Non-prunable parameters:    {non_prunable_params:,} ({non_prunable_params/encoder_params:.1%} of encoder)")
    print(f"   After pruning:")
    print(f"   - Kept parameters:            {remaining_params:,} (of prunable)")
    print(f"   - Pruned parameters:          {prunable_params_total - remaining_params:,} (of prunable)")
    print(f"   - Total remaining params:     {remaining_params + non_prunable_params:,}")
    print(f"   - Sparsity within prunable:   {1.0 - remaining_params / prunable_params_total:.2%}")
    print(f"   - Overall encoder sparsity:   {1.0 - (remaining_params + non_prunable_params) / encoder_params:.2%}")
    print(f"   - Target sparsity (config):   {pruning_config['weight']['sparsity']:.2%}")
    print("="*80)
    
    # -------------------------------
    # 8. Optional fine-tuning and evaluation
    # -------------------------------
    if hasattr(trainer, "evaluate"):
        print("\nStarting fine-tuning of the pruned model...")
        eval_results = trainer.evaluate()
        print("\nEvaluation Results:")
        print(f"  WER: {eval_results.get('eval_wer', 'N/A')}")
        print(f"  Loss: {eval_results.get('eval_loss', 'N/A')}")
        print(f"  Runtime: {eval_results.get('eval_runtime', 'N/A')}")
        print(f"  Samples per second: {eval_results.get('eval_samples_per_second', 'N/A')}")
        print(f"  Steps per second: {eval_results.get('eval_steps_per_second', 'N/A')}")
    else:
        print("\nTraining tools not available. Skipping fine-tuning and evaluation.")
    
    print("\nSNIP pruning example complete!")


if __name__ == "__main__":
    main() 
