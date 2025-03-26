"""
Optimization study implementation.
"""
import torch
import logging
import optuna
import dill
import json
from optuna.samplers import TPESampler
from copy import deepcopy

from chop import MaseGraph
import chop.passes as passes
from chop.models import CombinedWav2Vec2CTC

from optimization.pruning import apply_pruning
from optimization.smoothquant import apply_smoothquant
from optimization.quantization import apply_quantization
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
        study_name="wav2vec2_optimization",
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
    logger.info(f"  Quantization Method: {best_trial.user_attrs.get('quantization_method', 'N/A')}")
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
    """Save the best model from the study"""
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
    
    quant_methods = search_space["quantization"]["methods"]
    quant_method_idx = best_trial.params.get("quantization_method_idx", 0)
    quant_method_name, quant_class = quant_methods[quant_method_idx]
    
    # 1. Apply pruning
    logger.info(f"Applying pruning: {pruning_method}, sparsity: {pruning_sparsity}")
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
    
    # 4. Build bit config for quantization
    bit_config = None
    if quant_method_name != "full_precision":
        if quant_method_name == "integer":
            bit_config = {
                "weight_width": best_trial.params.get("weight_width"),
                "weight_frac_width": best_trial.params.get("weight_frac_width"),
                "data_in_width": best_trial.params.get("data_in_width"),
                "data_in_frac_width": best_trial.params.get("data_in_frac_width"),
                "bias_width": best_trial.params.get("bias_width"),
                "bias_frac_width": best_trial.params.get("bias_frac_width"),
            }
        elif quant_method_name in ["minifloat_denorm", "minifloat_ieee"]:
            bit_config = {
                "weight_width": best_trial.params.get("weight_width"),
                "weight_exponent_width": best_trial.params.get("weight_exponent_width"),
                "weight_exponent_bias": best_trial.params.get("weight_exponent_bias"),
                "data_in_width": best_trial.params.get("data_in_width"),
                "data_in_exponent_width": best_trial.params.get("data_in_exponent_width"),
                "data_in_exponent_bias": best_trial.params.get("data_in_exponent_bias"),
                "bias_width": best_trial.params.get("bias_width"),
                "bias_exponent_width": best_trial.params.get("bias_exponent_width"),
                "bias_exponent_bias": best_trial.params.get("bias_exponent_bias"),
            }
        else:
            bit_config = {
                "weight_width": best_trial.params.get("weight_width"),
                "data_in_width": best_trial.params.get("data_in_width"),
            }
    
    # 5. Apply quantization
    logger.info(f"Applying quantization: {quant_method_name}")
    quantized_model = apply_quantization(smoothed_mg.model, quant_method_name, quant_class, bit_config)
    
    # 6. Create final optimized model
    # Create final optimized model
    best_model = CombinedWav2Vec2CTC(
        encoder=quantized_model,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    
    # Save model's state dictionary instead of the full model
    model_filename = "best_optimized_model_state_dict.pt"
    torch.save(best_model.state_dict(), model_filename)
    
    # Save configuration
    config = {
        "pruning_method": pruning_method,
        "pruning_sparsity": pruning_sparsity,
        "structured_sparsity": structured_sparsity,
        "smoothquant_alpha": smoothquant_alpha,
        "quantization_method": quant_method_name,
        "bit_config": bit_config,
        "composite_score": best_trial.user_attrs.get("composite_score"),
    }
    
    with open("best_model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Best model state dict saved to {model_filename}")
    logger.info(f"Best model config saved to best_model_config.json")
    
    return best_model