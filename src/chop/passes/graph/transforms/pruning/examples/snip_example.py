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

from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms.pruning.prune import prune_transform_pass
from chop.passes.graph.transforms.pruning.snip_helper import SNIPCallback
import chop.passes as passes
from chop.passes.module import report_trainable_parameters_analysis_pass

# Check if transformers is available, otherwise give installation instructions
try:
    from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainerCallback
    from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
    from pyctcdecode import build_ctcdecoder
    transformers_available = True
except ImportError:
    transformers_available = False
    print("This example requires transformers, datasets, and pyctcdecode.")
    print("Please install them with: pip install transformers datasets pyctcdecode")

# Check if we have the necessary modules for training
try:
    from chop.tools import get_tokenized_dataset, get_trainer
    from chop.models import DataCollatorCTCWithPadding
    trainer_available = True
except ImportError:
    trainer_available = False
    print("Some tools for training are not available. Basic pruning will work, but not training.")

# Print available pruning methods
import chop.passes.graph.transforms.pruning.prune as prune_mod
importlib.reload(prune_mod)
print("Available pruning methods:", list(prune_mod.weight_criteria_map["local"]["elementwise"].keys()))

# -------------------------------
# Model and combined module definitions
# -------------------------------
class CombinedWav2Vec2CTC(nn.Module):
    def __init__(self, encoder, ctc_head, blank_id=0, beam_width=10, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.ctc_head = ctc_head
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.decoder = decoder  # Only used during inference

    def forward(self, input_values, attention_mask=None, labels=None):
        encoder_out = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = encoder_out["last_hidden_state"]
        logits = self.ctc_head(hidden_states)
        output = {"logits": logits, "labels": labels}

        if labels is not None:
            # Ensure labels has a batch dimension
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            batch_size, time_steps, _ = logits.shape
            input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=logits.device)
            target_lengths = (labels != -100).sum(dim=1)
            loss = F.ctc_loss(
                log_probs, labels, input_lengths, target_lengths,
                blank=self.blank_id, reduction="mean", zero_infinity=True
            )
            output["loss"] = loss
        else:
            if self.decoder is not None:
                log_probs = logits.log_softmax(dim=-1)
                log_probs_np = log_probs[0].cpu().detach().numpy()
                transcription = self.decoder.decode(log_probs_np, beam_width=self.beam_width).lower()
                output["transcription"] = transcription
        return output


def main():
    if not transformers_available:
        print("Required libraries not available. Exiting.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------
    # 1. Define the model and dataset
    # -------------------------------
    print("Loading model and dataset...")
    checkpoint = "facebook/wav2vec2-base-960h"
    tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
    dataset_name = "nyalpatel/condensed_librispeech_asr"
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(tokenizer_checkpoint)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    tokenizer = processor.tokenizer
    vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
    decoder = build_ctcdecoder(vocab)
    
    # Load a small subset of data
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
    
    # Load model
    model = AutoModelForCTC.from_pretrained(checkpoint)
    model.config.gradient_checkpointing = False
    encoder = model.wav2vec2    # static, FX-friendly
    ctc_head = model.lm_head    # dynamic CTC head, separate this
    
    # -------------------------------
    # 2. Define the MASE graph & metadata
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
    # 3. Print parameter count before pruning
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
    # 4. Apply SNIP callback for pruning preparation
    # -------------------------------
    print("\nPreparing for SNIP pruning...")
    
    if trainer_available:
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
    else:
        # Use direct monkey-patching without trainer
        print("Training tools not available, using direct SNIP computation...")
        
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
    # 5. Apply pruning pass using SNIP method
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
    # 6. Print parameter count after pruning
    # -------------------------------
    print("\nParameter count after pruning:")
    try:
        if hasattr(passes.module, "report_trainable_parameters_analysis_pass"):
            _, report_after = passes.module.report_trainable_parameters_analysis_pass(mg.model)
            for key, value in report_after.items():
                print(f"  {key}: {value}")
        else:
            raise AttributeError("report_trainable_parameters_analysis_pass not available")
    except (AttributeError, TypeError) as e:
        print(f"Could not use detailed parameter report: {e}")
        total_params = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
        print(f"  Total trainable parameters: {total_params}")
    
    # Count remaining parameters (non-pruned)
    remaining_params = 0
    for module in mg.model.modules():
        if hasattr(module, 'parametrizations'):
            for param_name, param_list in module.parametrizations.items():
                for param in param_list:
                    if hasattr(param, 'mask'):
                        remaining_params += param.mask.sum().item()
    
    total_params = sum(p.numel() for p in mg.model.parameters() if p.requires_grad)
    print(f"  Non-pruned parameters: {remaining_params}")
    print(f"  Actual sparsity: {1.0 - remaining_params / total_params:.4f}")
    
    # -------------------------------
    # 7. Optional fine-tuning and evaluation
    # -------------------------------
    if trainer_available:
        print("\nStarting fine-tuning of the pruned model...")
        trainer.train()
        
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