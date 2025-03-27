"""
Optimization study implementation with mixed precision support.
"""
import torch
import logging
import optuna
import json
from optuna.samplers import TPESampler
from copy import deepcopy

from chop import MaseGraph
import chop.passes as passes
from chop.models import CombinedWav2Vec2CTC
import torch.nn as nn

from optimization.pruning import apply_pruning
from optimization.smoothquant import apply_smoothquant
from optimization.quantization import apply_mixed_precision_quantization
from search.objective import objective
from config import NUM_TRIALS

# Set up logging
logger = logging.getLogger(__name__)

def run_optimization_study(baseline_model_data, n_trials=NUM_TRIALS):
    """Run the Optuna optimization study"""
    logger.info(f"Starting optimization study with {n_trials} trials...")
    
    # Create study with TPE sampler
    sampler = TPESampler()
    study = optuna.create_study(
        direction="maximize",
        study_name="wav2vec2_mixed_precision_optimization",
        sampler=sampler
    )
    
    study.optimize(
        lambda trial: objective(trial, baseline_model_data),
        n_trials=n_trials,
        timeout=60 * 60 * 12  # 12 hour timeout
    )
    
    # Get best trial
    best_trial = study.best_trial
    logger.info("\nBest trial:")
    logger.info(f"  Trial Number: {best_trial.number}")
    logger.info(f"  Composite Metric: {best_trial.value}")
    
    # Log mixed precision stats
    if "precision_decisions" in best_trial.user_attrs:
        precision_decisions = best_trial.user_attrs["precision_decisions"]
        type_counts = {}
        for layer_type in precision_decisions.values():
            type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
        
        logger.info("  Quantization Type Distribution:")
        for type_name, count in type_counts.items():
            logger.info(f"    {type_name}: {count} layers")
    
    logger.info(f"  Pruning Method: {best_trial.params.get('pruning_method', 'N/A')}")
    logger.info(f"  SmoothQuant Alpha: {best_trial.params.get('smoothquant_alpha', 'N/A')}")
    
    if "overall_sparsity" in best_trial.user_attrs:
        logger.info(f"  Overall Sparsity: {best_trial.user_attrs['overall_sparsity']:.2%}")
    
    logger.info(f"  Average Bit Width: {best_trial.user_attrs.get('avg_bitwidth', 'N/A')}")
    logger.info(f"  WER: {best_trial.user_attrs.get('runtime_average_wer', 'N/A')}")
    
    # Save best trial model
    save_best_model(best_trial, baseline_model_data)
    
    return study

def save_best_model(best_trial, baseline_model_data):
    """Save the best model from the study with mixed precision support"""
    logger.info("Saving best model...")
    
    # Reconstruct the best model
    mg_model = baseline_model_data["mg"].model
    ctc_head = baseline_model_data["ctc_head"]
    decoder = baseline_model_data["decoder"]
    search_space = baseline_model_data["search_space"]
    data_module = baseline_model_data["data_module"]
    checkpoint = baseline_model_data["checkpoint"]
    dataset_name = baseline_model_data["dataset_name"]
    
    # Get parameters from best trial
    pruning_method = best_trial.params.get("pruning_method")
    pruning_sparsity = best_trial.params.get("pruning_sparsity", 0.0)
    structured_sparsity = best_trial.params.get("structured_sparsity", False)
    smoothquant_alpha = best_trial.params.get("smoothquant_alpha")
    
    processor = baseline_model_data["processor"]

    # 1. Apply pruning
    logger.info(f"Applying pruning: {pruning_method}, sparsity: {pruning_sparsity}")

    if pruning_method in ["movement", "snip"]:
        # These methods need training data for initialization
        pruned_model = apply_pruning(
            mg_model, 
            pruning_method, 
            pruning_sparsity, 
            structured_sparsity,
            tokenized_dataset=baseline_model_data["tokenized_dataset"],
            tokenizer=baseline_model_data["tokenizer"],
            processor=processor,
            decoder=decoder,
            ctc_head=ctc_head
        )
    else:
        pruned_model = apply_pruning(mg_model, pruning_method, pruning_sparsity, structured_sparsity)
    
    # 2. Create MG for SmoothQuant
    pruned_mg = MaseGraph(
        pruned_model,
        hf_input_names=["input_values", "attention_mask"],
    )
    
    # Initialize metadata
    pruned_mg, _ = passes.init_metadata_analysis_pass(pruned_mg)
    
    # Create dummy input
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    
    # Add common metadata
    pruned_mg, _ = passes.add_common_metadata_analysis_pass(
        pruned_mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": False,
        }
    )
    
    # 3. Apply SmoothQuant
    logger.info(f"Applying SmoothQuant with alpha: {smoothquant_alpha}")
    smoothed_mg, _ = apply_smoothquant(pruned_mg, smoothquant_alpha, data_module, checkpoint, dataset_name)
    
    # 4. Apply mixed precision quantization
    # Define available precision choices
    precision_choices = [nn.Linear]  # Start with full precision
    
    # Add quantization types from search space
    for _, quant_class in search_space["quantization"]["methods"]:
        if quant_class not in precision_choices:
            precision_choices.append(quant_class)
    
    logger.info("Applying mixed precision quantization")
    quantized_model, precision_decisions = apply_mixed_precision_quantization(
        smoothed_mg.model, 
        precision_choices, 
        best_trial
    )
    
    # 5. Create final optimized model
    best_model = CombinedWav2Vec2CTC(
        encoder=quantized_model,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    
    # Save model's state dictionary instead of the full model
    model_filename = "best_mixed_precision_model_state_dict.pt"
    torch.save(best_model.state_dict(), model_filename)
    
    # Save configuration
    config = {
        "pruning_method": pruning_method,
        "pruning_sparsity": pruning_sparsity,
        "structured_sparsity": structured_sparsity,
        "smoothquant_alpha": smoothquant_alpha,
        "mixed_precision": True,
        "precision_decisions": precision_decisions,
        "composite_score": best_trial.user_attrs.get("new_composite_score"),
        "wer": best_trial.user_attrs.get("runtime_average_wer"),
        "latency": best_trial.user_attrs.get("runtime_average_latency"),
        "energy": best_trial.user_attrs.get("runtime_inference_energy_consumption"),
        "avg_bitwidth": best_trial.user_attrs.get("avg_bitwidth"),
    }
    
    with open("best_mixed_precision_model_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)  # Use default=str to handle non-serializable objects
    
    logger.info(f"Best model state dict saved to {model_filename}")
    logger.info(f"Best model config saved to best_mixed_precision_model_config.json")
    
    return best_model