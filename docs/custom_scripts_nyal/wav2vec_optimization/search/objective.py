"""
Objective functions for optimization with mixed precision support.
"""

import torch
import logging
from chop import MaseGraph
import chop.passes as passes
from chop.passes.graph import (
    runtime_analysis_pass,
    calculate_avg_bits_mg_analysis_pass,
)
from chop.models import CombinedWav2Vec2CTC
from chop.tools import get_trainer
import torch.nn as nn

from optimization.pruning import apply_pruning, calculate_pruning_metrics
from optimization.smoothquant import apply_smoothquant
from optimization.quantization import apply_mixed_precision_quantization
import config  # Import the config module instead of specific constants
from config import BATCH_SIZE  # We can keep BATCH_SIZE as direct import since it's not modified

# Set up logging
logger = logging.getLogger(__name__)

def objective(trial, baseline_model_data):
    """
    Objective function for optimization with mixed precision support that:
    1) Creates optimized model (mixed precision quantized + pruned)
    2) Trains the optimized model
    3) Performs runtime analysis
    4) Calculates composite score based on normalized metrics
    """
    # Unpack baseline data
    search_space = baseline_model_data["search_space"]
    mg = baseline_model_data["mg"]
    ctc_head = baseline_model_data["ctc_head"]
    decoder = baseline_model_data["decoder"]
    tokenized_dataset = baseline_model_data["tokenized_dataset"]
    tokenizer = baseline_model_data["tokenizer"]
    data_collator = baseline_model_data["data_collator"]
    data_module = baseline_model_data["data_module"]
    checkpoint = baseline_model_data["checkpoint"]
    dataset_name = baseline_model_data["dataset_name"]
    baseline_metrics = baseline_model_data["baseline_metrics"]
    processor = baseline_model_data["processor"]
    
    logger.info(f"Starting trial {trial.number}")
    logger.info(f"Using {config.EPOCHS} epochs for training")
    
    # -----------------------------
    # Phase 1: Pruning
    # -----------------------------
    pruning_method = trial.suggest_categorical("pruning_method", 
                                            search_space["pruning"]["methods"])

    sparsity = trial.suggest_categorical("pruning_sparsity", 
                                        search_space["pruning"]["sparsity_levels"])

    structured = False
    if pruning_method == "hwpq":
        structured = trial.suggest_categorical("structured_sparsity", 
                                            search_space["pruning"]["structured_options"])

    # Apply pruning to the model
    if pruning_method in ["movement", "snip"]:
        # These methods need training data for initialization
        pruned_model = apply_pruning(
            mg.model, 
            pruning_method, 
            sparsity, 
            structured,
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            processor=processor,
            decoder=decoder,
            ctc_head=ctc_head
        )
    else:
        pruned_model = apply_pruning(mg.model, pruning_method, sparsity, structured)

    # Calculate pruning metrics
    pruning_metrics = calculate_pruning_metrics(pruned_model)
    
    for k, v in pruning_metrics.items():
        trial.set_user_attr(k, v)
    
    # -----------------------------
    # Phase 2: SmoothQuant
    # -----------------------------
    
    # Create a new MG with the pruned model
    pruned_mg = MaseGraph(
        pruned_model,
        hf_input_names=["input_values", "attention_mask"],
    )
    
    # Initialize metadata for new graph
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
    
    # Select alpha for SmoothQuant
    alpha = trial.suggest_categorical("smoothquant_alpha", 
                                     search_space["smoothquant"]["alpha_values"])
    
    # Apply SmoothQuant
    smoothed_mg, _ = apply_smoothquant(pruned_mg, alpha, data_module, checkpoint, dataset_name)
    
    # -----------------------------
    # Phase 3: Mixed Precision Quantization
    # -----------------------------
    
    # Define available precision choices, including full precision
    precision_choices = [nn.Linear]  # Start with full precision
    
    # Add quantization types from search space
    for _, quant_class in search_space["quantization"]["methods"]:
        if quant_class not in precision_choices:
            precision_choices.append(quant_class)
    
    # Apply mixed precision quantization
    quantized_model, precision_decisions = apply_mixed_precision_quantization(
        smoothed_mg.model, 
        precision_choices, 
        trial
    )
    
    # Store the precision decisions
    trial.set_user_attr("precision_decisions", precision_decisions)
    
    # Count the number of layers for each quantization type
    type_counts = {}
    for layer_type in precision_decisions.values():
        type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
    
    for type_name, count in type_counts.items():
        trial.set_user_attr(f"count_{type_name}", count)
    
    # Create final optimized model combining all phases
    optimized_model = CombinedWav2Vec2CTC(
        encoder=quantized_model,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    
    # Store configuration info in trial
    trial.set_user_attr("mixed_precision", True)
    trial.set_user_attr("pruning_method", pruning_method)
    trial.set_user_attr("smoothquant_alpha", alpha)
    
    # Train the optimized model
    logger.info("Training optimized model...")
    trainer = get_trainer(
        model=optimized_model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        evaluate_metric="wer",
        data_collator=data_collator,
        num_train_epochs=config.EPOCHS,  # Use config.EPOCHS instead of EPOCHS
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
    )
    trainer.train()
    
    # Evaluate model
    eval_results = trainer.evaluate()
    wer = eval_results.get("eval_wer", None)
    loss = eval_results.get("eval_loss", None)
    trial.set_user_attr("eval_wer_pt", wer)
    trial.set_user_attr("eval_loss_pt", loss)

    trainer.model = trainer.model.cpu()
    
    # Create new MG for runtime analysis
    final_mg = MaseGraph(
        trainer.model.encoder,
        hf_input_names=["input_values", "attention_mask"],
    )

    # Initialize metadata
    final_mg, _ = passes.init_metadata_analysis_pass(final_mg)
    
    # Add common metadata
    final_mg, _ = passes.add_common_metadata_analysis_pass(
        final_mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": False,
        }
    )
    
    # Configure runtime analysis
    runtime_analysis_config = {
        "num_batches": 24,
        "num_GPU_warmup_batches": 2,
        "test": True,
        "data_module": data_module,
        "model": checkpoint,
        "accelerator": "cuda",
        "task": "ctc",
        "decoder": decoder,
        "beam_width": 10,
        "tokenizer": tokenizer,
        "batch_size": BATCH_SIZE,
        "sample_rate": 16000,
        "ctc_head": ctc_head,
    }
    
    # Run runtime analysis
    _, runtime_results = runtime_analysis_pass(final_mg, pass_args=runtime_analysis_config)
    trial.set_user_attr("runtime_metrics", runtime_results)

    # Run bit width analysis
    _, bitwidth_results = calculate_avg_bits_mg_analysis_pass(final_mg)
    data_avg_bit = bitwidth_results.get("data_avg_bit", 32)
    weight_avg_bit = bitwidth_results.get("w_avg_bit", 32)

    # Calculate the combined average bit width
    avg_bitwidth = (data_avg_bit + weight_avg_bit) / 2.0

    # Store both individual metrics and the combined average
    trial.set_user_attr("data_avg_bitwidth", data_avg_bit)
    trial.set_user_attr("weight_avg_bitwidth", weight_avg_bit)
    trial.set_user_attr("avg_bitwidth", avg_bitwidth)
        
    # Store individual runtime metrics
    relevant_keys = [
        "Average WER",
        "Average Latency",
        "Average RTF",
        "Average GPU Power Usage",
        "Inference Energy Consumption",
    ]
    
    for k in relevant_keys:
        val = runtime_results.get(k, float("inf"))
        trial.set_user_attr(f"runtime_{k.lower().replace(' ', '_')}", val)
    
    # Calculate percentage changes from baseline
    pct_changes = {}
    for k in relevant_keys:
        trial_val = runtime_results.get(k, None)
        base_val = baseline_metrics.get(k, None)
        if trial_val is not None and base_val is not None and base_val != 0:
            pct_change = (trial_val - base_val) / base_val * 100.0
            pct_changes[k] = pct_change
            trial.set_user_attr(f"pct_change_{k.lower().replace(' ', '_')}", pct_change)
    
    # Calculate bit width reduction
    base_bitwidth = baseline_metrics.get("avg_bitwidth", 32)
    bitwidth_reduction = (base_bitwidth - avg_bitwidth) / base_bitwidth
    trial.set_user_attr("bitwidth_reduction", bitwidth_reduction)
    
    # Normalized metrics calculation
    
    # 1. Cap WER at 1.0 (100%)
    wer_value = runtime_results.get("Average WER", 1)
    normalized_wer = min(wer_value, 1.0)
    trial.set_user_attr("normalized_wer", normalized_wer)
    
    # 2. Normalize latency between 0-1 based on percentage change
    latency_pct_change = pct_changes.get("Average Latency", 0.0)
    # Convert to 0-1 scale where 0 is best (large reduction), 1 is worst (large increase)
    normalized_latency = (latency_pct_change + 100) / 200
    normalized_latency = max(0.0, min(normalized_latency, 1.0))  # Clamp between 0-1
    trial.set_user_attr("normalized_latency", normalized_latency)
    
    # 3. Normalize energy between 0-1 based on percentage change
    energy_pct_change = pct_changes.get("Inference Energy Consumption", 0.0)
    normalized_energy = (energy_pct_change + 100) / 200
    normalized_energy = max(0.0, min(normalized_energy, 1.0))  # Clamp between 0-1
    trial.set_user_attr("normalized_energy", normalized_energy)
    
    # 4. Normalize bit width between 0-1 with max of 32
    normalized_bitwidth = avg_bitwidth / 32.0
    normalized_bitwidth = max(0.0, min(normalized_bitwidth, 1.0))  # Clamp between 0-1
    trial.set_user_attr("normalized_bitwidth", normalized_bitwidth)
    
    # 5. Sparsity is already normalized between 0-1
    sparsity = pruning_metrics["overall_sparsity"]
    
    # Define weights for each metric
    wer_weight = 0.4 
    latency_weight = 0.05
    energy_weight = 0.05
    bitwidth_weight = 0.2
    sparsity_weight = 0.2
    
    # Combine into new composite score
    composite_score = (
        wer_weight * normalized_wer +
        latency_weight * normalized_latency +
        energy_weight * normalized_energy +
        bitwidth_weight * normalized_bitwidth -
        sparsity_weight * sparsity  # Subtract because higher sparsity is better
    )
    
    # Invert for Optuna (which maximizes)
    composite_metric = -composite_score
    trial.set_user_attr("new_composite_score", composite_score)
    trial.set_user_attr("new_composite_metric", composite_metric)
    
    logger.info(f"Trial complete with new composite score: {composite_score}")
    logger.info(f"Normalized metrics: WER={normalized_wer:.4f}, Latency={normalized_latency:.4f}, "
                f"Energy={normalized_energy:.4f}, Bitwidth={normalized_bitwidth:.4f}, Sparsity={sparsity:.4f}")
    
    return composite_metric