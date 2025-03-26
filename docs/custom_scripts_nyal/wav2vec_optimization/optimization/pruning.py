"""
Pruning implementation for Wav2Vec2 optimization.
"""

import torch
import logging
import types
from copy import deepcopy
from chop import MaseGraph
import chop.passes as passes
from chop.passes.graph.transforms.pruning.snip_helper import SNIPCallback
from chop.passes.graph.transforms.pruning.prune_movment_helper import MovementTrackingCallback
from chop.tools import get_trainer
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC


# Set up logging
logger = logging.getLogger(__name__)

def apply_pruning(model, pruning_method, sparsity, structured_sparsity=False, 
                  tokenized_dataset=None, tokenizer=None, processor=None, 
                  decoder=None, ctc_head=None):
    """Apply pruning to the model based on specified parameters"""
    logger.info(f"Applying {pruning_method} pruning with sparsity {sparsity}")
    
    # Make a deep copy of the model and ensure it's on CPU
    pruned_model = deepcopy(model)
    pruned_model = pruned_model.cpu()
    
    # Ensure ctc_head is on CPU if provided
    if ctc_head is not None:
        ctc_head = ctc_head.cpu()
    
    # Create pruning config
    pruning_config = {
        "weight": {
            "sparsity": sparsity,
            "method": pruning_method,
            "scope": "local",
            "granularity": "elementwise",
        },
        "activation": {
            "sparsity": 0.0,
            "method": "random",
            "scope": "local",
            "granularity": "elementwise",
        },
    }
    
    # Special handling for movement pruning and SNIP if needed
    if pruning_method in ["movement", "snip"] and tokenized_dataset is not None:
        try:
            # Run specialized preprocessing in a separate function to isolate device issues
            if pruning_method == "movement":
                pruned_model = _apply_movement_preprocessing(
                    pruned_model, tokenized_dataset, tokenizer, processor, decoder, ctc_head
                )
            elif pruning_method == "snip":
                pruned_model = _apply_snip_preprocessing(
                    pruned_model, tokenized_dataset, tokenizer, processor, decoder, ctc_head
                )
                pruning_config["weight"]["importance"] = "snip_scores"
        except Exception as e:
            logger.warning(f"Error in specialized pruning preprocessing: {e}")
            logger.warning(f"Falling back to standard {pruning_method} pruning")
    
    # Explicitly put the model on CPU again after any preprocessing
    pruned_model = pruned_model.cpu()
    
    # Create MaseGraph with the model on CPU
    temp_mg = MaseGraph(
        pruned_model,
        hf_input_names=["input_values", "attention_mask"],
    )
    
    # Initialize metadata
    temp_mg, _ = passes.init_metadata_analysis_pass(temp_mg)
    
    # Create dummy input on CPU
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    
    # Make sure entire model graph is on CPU before adding metadata
    for module in temp_mg.model.modules():
        for param in module.parameters(recurse=False):
            if param.device.type != "cpu":
                param.data = param.data.cpu()
    
    # Add common metadata
    temp_mg, _ = passes.add_common_metadata_analysis_pass(
        temp_mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": True,  # Force device meta to avoid device mismatch
        }
    )
    
    # Apply pruning transform pass
    temp_mg, _ = passes.prune_transform_pass(temp_mg, pass_args=pruning_config)
    
    # Return the model, ensuring it's on CPU
    return temp_mg.model.cpu()

def _apply_movement_preprocessing(model, tokenized_dataset, tokenizer, processor, decoder, ctc_head):
    """Apply movement tracking preprocessing to collect importance scores"""
    logger.info("Applying movement tracking preprocessing")
    
    # Initialize all parameters on CPU
    model = model.cpu()
    if ctc_head is not None:
        ctc_head = ctc_head.cpu()
    
    # Initialize metadata for movement tracking
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and hasattr(module, "weight"):
            if not hasattr(module, "metadata"):
                module.metadata = {}
            # Initialize movement tracking with zeros on CPU
            module.metadata["weight"] = {"stats": {"movement": torch.zeros_like(module.weight, device="cpu")}}
    
    # Create combined model on CPU
    combined_model = CombinedWav2Vec2CTC(
        encoder=model,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    combined_model = combined_model.cpu()
    
    # Setup trainer with special configuration to avoid GPU
    trainer = get_trainer(
        model=combined_model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        evaluate_metric="wer",
        num_train_epochs=0.1,  # Just a short warm-up
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
    )
    # Override trainer's device to CPU
    trainer.args._n_gpu = 0  # Force CPU
    
    # Add movement tracking callback
    trainer.add_callback(MovementTrackingCallback())
    
    # Do warm-up training to collect movement data
    logger.info("Starting warm-up training for movement pruning...")
    trainer.train()
    logger.info("Warm-up training complete.")
    
    # Get the encoder with movement data and ensure it's on CPU
    return combined_model.encoder.cpu()

def _apply_snip_preprocessing(model, tokenized_dataset, tokenizer, processor, decoder, ctc_head):
    """Apply SNIP preprocessing to collect importance scores"""
    logger.info("Applying SNIP preprocessing")
    
    # Initialize all parameters on CPU
    model = model.cpu()
    if ctc_head is not None:
        ctc_head = ctc_head.cpu()
    
    # Create combined model on CPU
    combined_model = CombinedWav2Vec2CTC(
        encoder=model,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    combined_model = combined_model.cpu()
    
    # Setup trainer with special configuration to avoid GPU
    trainer = get_trainer(
        model=combined_model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        evaluate_metric="wer",
        num_train_epochs=0.1,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
    )
    # Override trainer's device to CPU
    trainer.args._n_gpu = 0  # Force CPU
    
    # Get a representative batch and ensure it's on CPU
    first_batch = next(iter(trainer.get_train_dataloader()))
    first_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in first_batch.items()}
    
    # Create and add SNIP callback
    snip_callback = SNIPCallback(representative_batch=first_batch)
    trainer.add_callback(snip_callback)
    
    # Run a minimal training step to trigger the callback
    trainer.train()
    
    # Get the encoder with SNIP scores and ensure it's on CPU
    return combined_model.encoder.cpu()

def calculate_pruning_metrics(model):
    """Calculate pruning metrics (sparsity, parameter counts)"""
    total_params = 0
    nonzero_params = 0
    pruned_params = 0
    
    # Count parameters using parametrizations (for masked models)
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            for p in module.parametrizations.weight:
                if hasattr(p, 'mask'):
                    total_in_layer = module.weight.numel()
                    nonzero_in_layer = p.mask.sum().item()
                    pruned_in_layer = total_in_layer - nonzero_in_layer
                    
                    pruned_params += pruned_in_layer
                    total_params += total_in_layer
    
    # If no pruning parametrizations found, count zeros in weights
    if total_params == 0:
        for name, param in model.named_parameters():
            if 'weight' in name and 'parametrizations' not in name:
                total_params += param.numel()
                nonzero_params += (param != 0).sum().item()
        
        pruned_params = total_params - nonzero_params
    else:
        nonzero_params = total_params - pruned_params
    
    # Calculate overall sparsity
    overall_sparsity = pruned_params / total_params if total_params > 0 else 0
    
    return {
        "total_weight_params": total_params,
        "nonzero_weight_params": nonzero_params,
        "pruned_weight_params": pruned_params,
        "overall_sparsity": overall_sparsity
    }