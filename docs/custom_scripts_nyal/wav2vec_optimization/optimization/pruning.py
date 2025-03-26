"""
Pruning implementation for Wav2Vec2 optimization.
"""

import torch
import logging
from copy import deepcopy
from chop import MaseGraph
import chop.passes as passes

# Set up logging
logger = logging.getLogger(__name__)

"""
Pruning implementation for Wav2Vec2 optimization.
"""

import torch
import logging
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
    
    # Make a copy of the model
    pruned_model = deepcopy(model)
    pruned_model = pruned_model.cpu()
    
    # Create pruning config
    pruning_config = {
        "weight": {
            "sparsity": sparsity,
            "method": pruning_method,
            "scope": "local",
            "structured_sparsity": structured_sparsity
        },
        "activation": {
            "sparsity": 0.0,
            "method": "random",
            "scope": "local",
        },
    }
    
    # Create temporary MaseGraph for this model instance
    temp_mg = MaseGraph(
        pruned_model,
        hf_input_names=["input_values", "attention_mask"],
    )
    
    # Initialize metadata
    temp_mg, _ = passes.init_metadata_analysis_pass(temp_mg)
    
    # Create dummy input
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    
    # Add common metadata
    temp_mg, _ = passes.add_common_metadata_analysis_pass(
        temp_mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": False,
        }
    )
    
    # Special handling for movement pruning and SNIP
    if pruning_method in ["movement", "snip"] and tokenized_dataset is not None:
        # For movement pruning, we need to do warm-up training to collect movement data
        if pruning_method == "movement":
            # Initialize metadata for movement tracking
            for module in temp_mg.model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and hasattr(module, "weight"):
                    if not hasattr(module, "metadata"):
                        module.metadata = {}
                    module.metadata["weight"] = {"stats": {"movement": torch.zeros_like(module.weight)}}
            
            # Create combined model
            combined_model = CombinedWav2Vec2CTC(
                encoder=temp_mg.model,
                ctc_head=ctc_head,
                decoder=decoder,
                beam_width=10
            )
            
            # Setup trainer
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
            
            # Add movement tracking callback
            trainer.add_callback(MovementTrackingCallback())
            
            # Do warm-up training to collect movement data
            logger.info("Starting warm-up training for movement pruning...")
            trainer.train()
            logger.info("Warm-up training complete.")
            
            # Get the updated model with movement tracking data
            temp_mg.model = combined_model.encoder
            
        elif pruning_method == "snip":
            # Create combined model
            combined_model = CombinedWav2Vec2CTC(
                encoder=temp_mg.model,
                ctc_head=ctc_head,
                decoder=decoder,
                beam_width=10
            )
            
            # Setup trainer to get dataloader
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
            
            # Get representative batch for SNIP
            first_batch = next(iter(trainer.get_train_dataloader()))
            
            # Use SNIPCallback to prepare the model for SNIP pruning
            snip_callback = SNIPCallback(representative_batch=first_batch)
            snip_callback.on_init_end(trainer)
            
            # Get the updated model with SNIP weights
            temp_mg.model = combined_model.encoder
    
    # Apply pruning transform pass
    temp_mg, _ = passes.prune_transform_pass(temp_mg, pass_args=pruning_config)

    return temp_mg.model

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
