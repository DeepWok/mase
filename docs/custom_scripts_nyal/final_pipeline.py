import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import dill
from copy import deepcopy
from pathlib import Path
from transformers import AutoModelForCTC, Wav2Vec2Processor
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_trainer, get_tokenized_dataset
from chop.tools.utils import deepsetattr
from chop.models import DataCollatorCTCWithPadding, CombinedWav2Vec2CTC
from pyctcdecode import build_ctcdecoder
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    runtime_analysis_pass,
    onnx_runtime_interface_pass,
    quantize_transform_pass,
    prune_transform_pass,
    bit_width_analysis_pass,
)
from chop.dataset.nlp.speech_recognition import CondensedLibrispeechASRDataset
from chop.dataset import MaseDataModule
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------
# 1. Model Importing and Dataset Setup
# -------------------------------

def import_model_and_dataset():
    """Import model, tokenizer, and dataset"""
    logger.info("Importing model and dataset...")
    
    checkpoint = "facebook/wav2vec2-base-960h"
    tokenizer_checkpoint = "facebook/wav2vec2-base-960h"
    dataset_name = "librispeech_asr"
    
    # Get tokenized dataset, tokenizer, and processor
    tokenized_dataset, tokenizer, processor = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
        return_processor=True,
    )
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    vocab = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
    decoder = build_ctcdecoder(vocab)
    
    # Load model
    model = AutoModelForCTC.from_pretrained(checkpoint)
    model.config.gradient_checkpointing = True
    encoder = model.wav2vec2  # static, FX-friendly
    ctc_head = model.lm_head    # dynamic CTC head, separate this
    
    # Setup data module
    dataset_path = Path("./preprocessed_data")
    condensed_dataset = CondensedLibrispeechASRDataset(dataset_path=dataset_path, split="train")
    condensed_dataset.prepare_data()
    condensed_dataset.setup()
    
    data_module = MaseDataModule(
        name="nyalpatel/condensed_librispeech_asr",
        batch_size=1,
        model_name=checkpoint,
        num_workers=0,
    )
    data_module.prepare_data()
    data_module.setup()
    
    logger.info("Model and dataset imported successfully")
    
    return {
        "encoder": encoder,
        "ctc_head": ctc_head,
        "tokenized_dataset": tokenized_dataset,
        "tokenizer": tokenizer,
        "processor": processor,
        "data_collator": data_collator,
        "vocab": vocab,
        "decoder": decoder,
        "data_module": data_module,
        "checkpoint": checkpoint,
        "dataset_name": dataset_name,
    }

# -------------------------------
# 2. Initial MASE Graph Setup
# -------------------------------

class ONNXWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder 

    def forward(self, inputs):
        if isinstance(inputs, dict):
            input_values = inputs["input_values"]
            attention_mask = inputs["attention_mask"]
        else:
            input_values = inputs
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
        return self.encoder(input_values, attention_mask=attention_mask)
    
    @property
    def graph(self):
        # Expose the underlying FX graph for later passes
        return self.encoder.graph

def setup_mase_graph(encoder):
    """Create and initialize MASE graph with metadata"""
    logger.info("Setting up MASE graph...")
    
    # Create MASE graph
    mg = MaseGraph(
        encoder,
        hf_input_names=["input_values", "attention_mask"],
    )
    
    # Initialize metadata
    mg, _ = passes.init_metadata_analysis_pass(mg)
    
    # Create dummy input for analysis
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    
    # Add common metadata
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": False,
        }
    )
    
    logger.info("MASE graph setup complete")
    
    return mg, dummy_in

# -------------------------------
# 3. Create Combined Model
# -------------------------------

def create_combined_model(mg, ctc_head, decoder):
    """Create a combined model with encoder, CTC head, and decoder"""
    logger.info("Creating combined model...")
    
    combined_model = CombinedWav2Vec2CTC(
        encoder=mg.model,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    
    logger.info("Combined model created successfully")
    
    return combined_model

# -------------------------------
# 4. Run Baseline Metrics
# -------------------------------

def run_baseline_metrics(mg, data_module, checkpoint, dataset_name, decoder, tokenizer):
    """Run baseline metrics to establish reference performance"""
    logger.info("Running baseline metrics...")
    
    # Wrap the model for ONNX compatibility
    mg.model = ONNXWrapper(mg.model)
    
    # Configure SmoothQuant with alpha=0 (no smoothing)
    smoothquant_config = {
        "smoothquant": True,
        "alpha": 0.0,
        "model": checkpoint,
        "task": "ctc",
        "dataset": dataset_name,
        "accelerator": "cuda",
        "data_module": data_module,
        "batch_size": 1,
    }
    
    # Configure runtime analysis
    runtime_analysis_config = {
        "num_batches": 100,
        "num_GPU_warmup_batches": 5,
        "test": True,
        "data_module": data_module,
        "model": checkpoint,
        "accelerator": "cuda",
        "task": "ctc",
        "decoder": decoder,
        "beam_width": 10,
        "tokenizer": tokenizer,
        "batch_size": 2,
        "sample_rate": 16000,
    }
    
    # Run ONNX interface pass and runtime analysis
    mg, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=smoothquant_config)
    _, baseline_results = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
    
    # Run bit width analysis for baseline
    _, bitwidth_results = bit_width_analysis_pass(mg)
    baseline_results.update({"avg_bitwidth": bitwidth_results.get("average_bitwidth", 32)})
    
    logger.info("Baseline metrics complete")
    logger.info("Baseline Results:")
    for k, v in baseline_results.items():
        logger.info(f"  {k}: {v}")
    
    return baseline_results, mg

# -------------------------------
# 5. Define Search Space
# -------------------------------

def define_search_space():
    """Define the search space for optimization"""
    logger.info("Defining search space...")
    
    # Quantization methods
    quantization_methods = [
        ("full_precision", nn.Linear),
        ("integer", LinearInteger),
        ("minifloat_denorm", LinearMinifloatDenorm),
        ("minifloat_ieee", LinearMinifloatIEEE),
        ("log", LinearLog),
        ("block_fp", LinearBlockFP),
    ]
    
    # Pruning methods
    pruning_methods = [
        "none",
        "hwpq",
        "magnitude",
        "random",
    ]
    
    # Width and fractional width choices
    width_choices = [8, 16, 32]
    frac_width_choices = [4, 8, 16]
    
    # SmoothQuant alpha choices
    alpha_choices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Sparsity choices
    sparsity_choices = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Structured sparsity choices
    structured_sparsity_choices = [True, False]
    
    search_space = {
        "quantization_methods": quantization_methods,
        "pruning_methods": pruning_methods,
        "width_choices": width_choices,
        "frac_width_choices": frac_width_choices,
        "alpha_choices": alpha_choices,
        "sparsity_choices": sparsity_choices,
        "structured_sparsity_choices": structured_sparsity_choices,
    }
    
    logger.info("Search space defined")
    
    return search_space

# -------------------------------
# 6. Pruning and Sparsity Metrics
# -------------------------------

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

# -------------------------------
# 7. Model Transformation Functions
# -------------------------------

def apply_pruning(model, trial, pruning_method):
    """Apply pruning to the model based on trial parameters"""
    if pruning_method == "none":
        return model
    
    # Make a copy of the model
    pruned_model = deepcopy(model)
    
    # Select sparsity level
    sparsity = trial.suggest_categorical("pruning_sparsity", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Use unstructured sparsity by default, but allow structured for hwpq
    structured_sparsity = False
    if pruning_method == "hwpq":
        structured_sparsity = trial.suggest_categorical("structured_sparsity", [True, False])
    
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
    
    # Apply pruning transform pass
    temp_mg, _ = passes.prune_transform_pass(temp_mg, pass_args=pruning_config)
    
    return temp_mg.model

def apply_quantization(model, trial, quant_method, quant_class):
    """Apply quantization to the model based on trial parameters"""
    if quant_method == "full_precision":
        return model
    
    # Make a copy of the model
    quant_model = deepcopy(model)
    
    # Apply quantization to each linear layer
    for name, layer in quant_model.named_modules():
        if isinstance(layer, nn.Linear):
            kwargs = {"in_features": layer.in_features, "out_features": layer.out_features}
            
            if quant_method == "integer":
                config = {
                    "weight_width": trial.suggest_categorical(f"{name[:10]}_weight_width", [8, 16, 32]),
                    "weight_frac_width": trial.suggest_categorical(f"{name[:10]}_weight_frac_width", [4, 8, 16]),
                    "data_in_width": trial.suggest_categorical(f"{name[:10]}_data_in_width", [8, 16, 32]),
                    "data_in_frac_width": trial.suggest_categorical(f"{name[:10]}_data_in_frac_width", [4, 8, 16]),
                    "bias_width": trial.suggest_categorical(f"{name[:10]}_bias_width", [8, 16, 32]),
                    "bias_frac_width": trial.suggest_categorical(f"{name[:10]}_bias_frac_width", [4, 8, 16]),
                    "floor": False,
                }
            elif quant_method in ["minifloat_denorm", "minifloat_ieee"]:
                config = {
                    "weight_width": trial.suggest_categorical(f"{name[:10]}_weight_width", [8, 16, 32]),
                    "weight_exponent_width": 5,
                    "weight_exponent_bias": 15,
                    "data_in_width": trial.suggest_categorical(f"{name[:10]}_data_in_width", [8, 16, 32]),
                    "data_in_exponent_width": 5,
                    "data_in_exponent_bias": 15,
                    "bias_width": trial.suggest_categorical(f"{name[:10]}_bias_width", [8, 16, 32]),
                    "bias_exponent_width": 5,
                    "bias_exponent_bias": 15,
                }
            else:
                # Default config for other quantization methods
                config = {
                    "weight_width": trial.suggest_categorical(f"{name[:10]}_weight_width", [8, 16, 32]),
                    "data_in_width": trial.suggest_categorical(f"{name[:10]}_data_in_width", [8, 16, 32]),
                }
            
            # Create new quantized layer
            new_layer = quant_class(**kwargs, config=config)
            new_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()
            
            # Replace the original layer with the quantized one
            deepsetattr(quant_model, name, new_layer)
    
    return quant_model

def construct_optimized_model(trial, mg_model, ctc_head, decoder, baseline_model_data):
    """
    Construct a model with both quantization and pruning applied
    based on trial parameters
    """
    logger.info("Constructing optimized model for trial...")
    
    # Select quantization method
    quant_methods = baseline_model_data["search_space"]["quantization_methods"]
    quant_method_idx = trial.suggest_categorical("quantization_method_idx", list(range(len(quant_methods))))
    quant_method_name, quant_class = quant_methods[quant_method_idx]
    
    # Select pruning method
    pruning_methods = baseline_model_data["search_space"]["pruning_methods"]
    pruning_method = trial.suggest_categorical("pruning_method", pruning_methods)
    
    # Make a copy of the base encoder
    encoder_copy = deepcopy(mg_model)
    
    # Apply pruning first (if enabled)
    if pruning_method != "none":
        encoder_copy = apply_pruning(encoder_copy, trial, pruning_method)
    
    # Apply quantization (if not full precision)
    if quant_method_name != "full_precision":
        encoder_copy = apply_quantization(encoder_copy, trial, quant_method_name, quant_class)
    
    # Create the combined model
    optimized_model = CombinedWav2Vec2CTC(
        encoder=encoder_copy,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    
    # Calculate and store pruning metrics if pruning was applied
    if pruning_method != "none":
        pruning_metrics = calculate_pruning_metrics(optimized_model.encoder)
        for k, v in pruning_metrics.items():
            trial.set_user_attr(k, v)
    
    trial.set_user_attr("quantization_method", quant_method_name)
    trial.set_user_attr("pruning_method", pruning_method)
    
    logger.info(f"Optimized model constructed with quantization={quant_method_name}, pruning={pruning_method}")
    
    return optimized_model

# -------------------------------
# 8. Objective Function
# -------------------------------

def objective(trial, baseline_model_data):
    """
    Objective function for optimization that:
    1) Creates optimized model (quantized + pruned)
    2) Trains the optimized model
    3) Performs runtime analysis
    4) Calculates composite score based on metrics
    """
    # Unpack baseline data
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
    
    # 1. Construct optimized model
    optimized_model = construct_optimized_model(
        trial, 
        mg.model, 
        ctc_head, 
        decoder, 
        baseline_model_data
    )
    
    # 2. Train the optimized model
    logger.info("Training optimized model...")
    trainer = get_trainer(
        model=optimized_model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        evaluate_metric="wer",
        data_collator=data_collator,
        num_train_epochs=1,  # Reduced for faster trials
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
    
    # 3. Prepare for runtime analysis
    logger.info("Running performance analysis...")
    
    # Select alpha parameter for SmoothQuant
    alpha_val = trial.suggest_categorical("smoothquant_alpha", baseline_model_data["search_space"]["alpha_choices"])
    
    # Set up model for ONNX
    mg.model = trainer.model.encoder
    mg.model = ONNXWrapper(mg.model)
    
    # SmoothQuant configuration
    smoothquant_config = {
        "smoothquant": True,
        "alpha": alpha_val,
        "model": checkpoint,
        "task": "ctc",
        "dataset": dataset_name,
        "accelerator": "cuda",
        "data_module": data_module,
        "batch_size": 1,
    }
    
    # Runtime analysis configuration
    runtime_analysis_config = {
        "num_batches": 100,
        "num_GPU_warmup_batches": 5,
        "test": True,
        "data_module": data_module,
        "model": checkpoint,
        "accelerator": "cuda",
        "task": "ctc",
        "decoder": decoder,
        "beam_width": 10,
        "tokenizer": tokenizer,
        "batch_size": 2,
        "sample_rate": 16000,
    }
    
    # Run ONNX interface pass
    mg, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=smoothquant_config)
    
    # Run runtime analysis
    _, runtime_results = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
    trial.set_user_attr("runtime_metrics", runtime_results)
    
    # Run bit width analysis
    _, bitwidth_results = bit_width_analysis_pass(mg)
    avg_bitwidth = bitwidth_results.get("average_bitwidth", 32)
    trial.set_user_attr("avg_bitwidth", avg_bitwidth)
    
    # 4. Calculate metrics and composite score
    relevant_keys = [
        "Average WER",
        "Average Latency",
        "Average RTF",
        "Average GPU Power Usage",
        "Inference Energy Consumption",
    ]
    
    # Store individual metrics
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
    
    # Calculate sparsity if pruning was applied
    sparsity = trial.user_attrs.get("overall_sparsity", 0.0)
    
    # Calculate composite score based on WER, latency, energy, bit width, and sparsity
    # Lower WER, latency, energy, bit width are better
    # Higher sparsity is better
    
    # Get the percentage changes (negative means improvement)
    wer_change = pct_changes.get("Average WER", 0.0)
    latency_change = pct_changes.get("Average Latency", 0.0)
    energy_change = pct_changes.get("Inference Energy Consumption", 0.0)
    
    # Weight the different metrics (these can be tuned)
    wer_weight = 0.4  # Accuracy is important
    latency_weight = 0.2
    energy_weight = 0.1
    bitwidth_weight = 0.2
    sparsity_weight = 0.1
    
    # Combine into composite score (negative is better as it means reduction)
    composite_score = (
        wer_weight * wer_change +
        latency_weight * latency_change +
        energy_weight * energy_change -
        bitwidth_weight * (bitwidth_reduction * 100) -  # Convert to percentage
        sparsity_weight * (sparsity * 100)  # Convert to percentage
    )
    
    # Invert for Optuna (which maximizes)
    composite_metric = -composite_score
    
    trial.set_user_attr("composite_score", composite_score)
    trial.set_user_attr("composite_metric", composite_metric)
    
    logger.info(f"Trial complete with composite score: {composite_score}")
    
    return composite_metric

# -------------------------------
# 9. Optimization Study
# -------------------------------

def run_optimization_study(baseline_model_data, n_trials=30):
    """Run the Optuna optimization study"""
    logger.info(f"Starting optimization study with {n_trials} trials...")
    
    # Create study with TPE sampler
    sampler = TPESampler()
    study = optuna.create_study(
        direction="maximize",
        study_name="wav2vec2_optimization",
        sampler=sampler
    )
    
    # Run optimization
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
    logger.info(f"  Pruning Method: {best_trial.user_attrs.get('pruning_method', 'N/A')}")
    
    if "overall_sparsity" in best_trial.user_attrs:
        logger.info(f"  Overall Sparsity: {best_trial.user_attrs['overall_sparsity']:.2%}")
    
    logger.info(f"  Average Bit Width: {best_trial.user_attrs.get('avg_bitwidth', 'N/A')}")
    logger.info(f"  WER: {best_trial.user_attrs.get('runtime_average_wer', 'N/A')}")
    logger.info(f"  Latency: {best_trial.user_attrs.get('runtime_average_latency', 'N/A')}")
    
    # Save best trial model
    save_best_model(best_trial, baseline_model_data)
    
    # Process and analyze study results
    results_df = process_study_results(study)
    
    # Create visualizations
    create_visualizations(study, results_df, baseline_model_data["baseline_metrics"])
    
    return study, results_df

# -------------------------------
# 10. Results Processing and Visualization
# -------------------------------
def process_study_results(study):
    """Process study results into a DataFrame"""
    logger.info("Processing study results...")
    
    results = []
    for t in sorted(study.trials, key=lambda t: t.number):
        # Basic trial info
        row = {
            "trial_number": t.number,
            "composite_metric": t.value,
            "composite_score": t.user_attrs.get("composite_score", None),
        }
        
        # Trial parameters
        row.update({
            "quantization_method": t.user_attrs.get("quantization_method", "N/A"),
            "pruning_method": t.user_attrs.get("pruning_method", "N/A"),
            "smoothquant_alpha": t.params.get("smoothquant_alpha", None),
        })
        
        # If pruning was used, add pruning parameters
        if t.user_attrs.get("pruning_method", "none") != "none":
            row.update({
                "pruning_sparsity": t.params.get("pruning_sparsity", None),
                "structured_sparsity": t.params.get("structured_sparsity", None),
                "overall_sparsity": t.user_attrs.get("overall_sparsity", None),
            })
        
        # Performance metrics
        runtime_metrics = ["average_wer", "average_latency", "average_rtf", 
                          "average_gpu_power_usage", "inference_energy_consumption"]
        
        for metric in runtime_metrics:
            row[f"runtime_{metric}"] = t.user_attrs.get(f"runtime_{metric}", None)
            row[f"pct_change_{metric}"] = t.user_attrs.get(f"pct_change_{metric}", None)
        
        # Bit width metrics
        row["avg_bitwidth"] = t.user_attrs.get("avg_bitwidth", None)
        row["bitwidth_reduction"] = t.user_attrs.get("bitwidth_reduction", None)
        
        # Add pytorch evaluation metrics
        row["eval_wer_pt"] = t.user_attrs.get("eval_wer_pt", None)
        row["eval_loss_pt"] = t.user_attrs.get("eval_loss_pt", None)
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_name = "optuna_study_results.csv"
    df.to_csv(csv_name, index=False)
    logger.info(f"Study results saved to {csv_name}")
    
    return df

def save_best_model(best_trial, baseline_model_data):
    """Save the best model from the study"""
    logger.info("Saving best model...")
    
    # Reconstruct the best model
    best_model = construct_optimized_model(
        best_trial, 
        baseline_model_data["mg"].model, 
        baseline_model_data["ctc_head"], 
        baseline_model_data["decoder"],
        baseline_model_data
    )
    
    # Save model using dill
    model_filename = "best_optimized_model.pkl"
    with open(model_filename, "wb") as f:
        dill.dump(best_model, f)
    
    logger.info(f"Best model saved to {model_filename}")
    
    return best_model

def create_visualizations(study, results_df, baseline_metrics):
    """Create visualizations for the optimization study"""
    logger.info("Creating visualizations...")
    
    # 1. Optimization history plot
    plt.figure(figsize=(12, 6))
    plt.title("Optimization History")
    plt.plot(results_df["trial_number"], results_df["composite_metric"], "o-")
    plt.xlabel("Trial Number")
    plt.ylabel("Composite Metric (higher is better)")
    plt.grid(True)
    plt.savefig("optimization_history.png")
    plt.close()
    
    # 2. Parallel coordinate plot for hyperparameters
    try:
        optuna.visualization.plot_parallel_coordinate(
            study, params=["quantization_method_idx", "pruning_method", "smoothquant_alpha"]
        )
        plt.savefig("parallel_coordinate.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create parallel coordinate plot: {e}")
    
    # 3. Parameter importance
    try:
        optuna.visualization.plot_param_importances(study)
        plt.savefig("param_importances.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create parameter importance plot: {e}")
    
    # 4. WER vs Latency scatter plot
    if "runtime_average_wer" in results_df.columns and "runtime_average_latency" in results_df.columns:
        plt.figure(figsize=(10, 8))
        
        # Color points by quantization method
        quant_methods = results_df["quantization_method"].unique()
        cm = plt.cm.get_cmap("tab10", len(quant_methods))
        
        for i, method in enumerate(quant_methods):
            method_df = results_df[results_df["quantization_method"] == method]
            plt.scatter(
                method_df["runtime_average_latency"], 
                method_df["runtime_average_wer"],
                label=method,
                alpha=0.7,
                c=[cm(i)],
                s=50
            )
        
        # Add baseline
        if "Average WER" in baseline_metrics and "Average Latency" in baseline_metrics:
            plt.scatter(
                baseline_metrics["Average Latency"],
                baseline_metrics["Average WER"],
                marker="*",
                s=200,
                c="red",
                label="Baseline"
            )
        
        plt.xlabel("Latency (ms)")
        plt.ylabel("WER (%)")
        plt.title("WER vs Latency Trade-off")
        plt.legend(title="Quantization Method")
        plt.grid(True)
        plt.savefig("wer_vs_latency.png")
        plt.close()
    
    # 5. Sparsity vs performance metrics
    if "overall_sparsity" in results_df.columns:
        sparsity_df = results_df[results_df["overall_sparsity"].notnull()]
        
        if not sparsity_df.empty:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # WER vs Sparsity
            if "runtime_average_wer" in sparsity_df.columns:
                axs[0].scatter(sparsity_df["overall_sparsity"], sparsity_df["runtime_average_wer"])
                axs[0].set_xlabel("Sparsity")
                axs[0].set_ylabel("WER (%)")
                axs[0].set_title("WER vs Sparsity")
                axs[0].grid(True)
            
            # Latency vs Sparsity
            if "runtime_average_latency" in sparsity_df.columns:
                axs[1].scatter(sparsity_df["overall_sparsity"], sparsity_df["runtime_average_latency"])
                axs[1].set_xlabel("Sparsity")
                axs[1].set_ylabel("Latency (ms)")
                axs[1].set_title("Latency vs Sparsity")
                axs[1].grid(True)
            
            # Energy vs Sparsity
            if "runtime_inference_energy_consumption" in sparsity_df.columns:
                axs[2].scatter(sparsity_df["overall_sparsity"], sparsity_df["runtime_inference_energy_consumption"])
                axs[2].set_xlabel("Sparsity")
                axs[2].set_ylabel("Energy Consumption")
                axs[2].set_title("Energy vs Sparsity")
                axs[2].grid(True)
            
            plt.tight_layout()
            plt.savefig("sparsity_performance.png")
            plt.close()
    
    # 6. Bit width vs performance metrics
    if "avg_bitwidth" in results_df.columns:
        bitwidth_df = results_df[results_df["avg_bitwidth"].notnull()]
        
        if not bitwidth_df.empty:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # WER vs Bit Width
            if "runtime_average_wer" in bitwidth_df.columns:
                axs[0].scatter(bitwidth_df["avg_bitwidth"], bitwidth_df["runtime_average_wer"])
                axs[0].set_xlabel("Average Bit Width")
                axs[0].set_ylabel("WER (%)")
                axs[0].set_title("WER vs Bit Width")
                axs[0].grid(True)
            
            # Latency vs Bit Width
            if "runtime_average_latency" in bitwidth_df.columns:
                axs[1].scatter(bitwidth_df["avg_bitwidth"], bitwidth_df["runtime_average_latency"])
                axs[1].set_xlabel("Average Bit Width")
                axs[1].set_ylabel("Latency (ms)")
                axs[1].set_title("Latency vs Bit Width")
                axs[1].grid(True)
            
            # Energy vs Bit Width
            if "runtime_inference_energy_consumption" in bitwidth_df.columns:
                axs[2].scatter(bitwidth_df["avg_bitwidth"], bitwidth_df["runtime_inference_energy_consumption"])
                axs[2].set_xlabel("Average Bit Width")
                axs[2].set_ylabel("Energy Consumption")
                axs[2].set_title("Energy vs Bit Width")
                axs[2].grid(True)
            
            plt.tight_layout()
            plt.savefig("bitwidth_performance.png")
            plt.close()
    
    # 7. Method comparison box plots
    # Create plots comparing different quantization methods
    plt.figure(figsize=(12, 8))
    
    # Compare WER across methods
    plt.subplot(2, 2, 1)
    if "quantization_method" in results_df.columns and "runtime_average_wer" in results_df.columns:
        results_df.boxplot(column="runtime_average_wer", by="quantization_method", ax=plt.gca())
        plt.ylabel("WER (%)")
        plt.title("WER by Quantization Method")
        plt.suptitle("")  # Remove pandas-generated suptitle
        plt.xticks(rotation=45)
    
    # Compare Latency across methods
    plt.subplot(2, 2, 2)
    if "quantization_method" in results_df.columns and "runtime_average_latency" in results_df.columns:
        results_df.boxplot(column="runtime_average_latency", by="quantization_method", ax=plt.gca())
        plt.ylabel("Latency (ms)")
        plt.title("Latency by Quantization Method")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare Energy across methods
    plt.subplot(2, 2, 3)
    if "quantization_method" in results_df.columns and "runtime_inference_energy_consumption" in results_df.columns:
        results_df.boxplot(column="runtime_inference_energy_consumption", by="quantization_method", ax=plt.gca())
        plt.ylabel("Energy Consumption")
        plt.title("Energy by Quantization Method")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare bit width across methods
    plt.subplot(2, 2, 4)
    if "quantization_method" in results_df.columns and "avg_bitwidth" in results_df.columns:
        results_df.boxplot(column="avg_bitwidth", by="quantization_method", ax=plt.gca())
        plt.ylabel("Average Bit Width")
        plt.title("Bit Width by Quantization Method")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("method_comparison.png")
    plt.close()
    
    logger.info("Visualizations created")

# -------------------------------
# 11. Main Function
# -------------------------------

def main():
    """Main function to run the optimization pipeline"""
    logger.info("Starting optimization pipeline")
    
    # 1. Import model and dataset
    model_data = import_model_and_dataset()
    
    # 2. Setup MASE graph
    mg, dummy_in = setup_mase_graph(model_data["encoder"])
    model_data["mg"] = mg
    
    # 3. Create combined model
    combined_model = create_combined_model(mg, model_data["ctc_head"], model_data["decoder"])
    
    # 4. Run baseline metrics
    baseline_metrics, updated_mg = run_baseline_metrics(
        mg, 
        model_data["data_module"], 
        model_data["checkpoint"], 
        model_data["dataset_name"], 
        model_data["decoder"], 
        model_data["tokenizer"]
    )
    model_data["mg"] = updated_mg
    
    # 5. Define search space
    search_space = define_search_space()
    
    # Prepare baseline_model_data dictionary for optimization
    baseline_model_data = {
        **model_data,
        "baseline_metrics": baseline_metrics,
        "search_space": search_space,
    }
    
    # 6. Run optimization study
    study, results_df = run_optimization_study(baseline_model_data, n_trials=30)
    
    # 7. Save final results
    logger.info("Optimization pipeline complete")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best composite metric: {study.best_trial.value}")
    
    # Create a summary of the best configuration
    best_config = {
        "quantization_method": study.best_trial.user_attrs.get("quantization_method", "N/A"),
        "pruning_method": study.best_trial.user_attrs.get("pruning_method", "N/A"),
        "smoothquant_alpha": study.best_trial.params.get("smoothquant_alpha", None),
        "overall_sparsity": study.best_trial.user_attrs.get("overall_sparsity", None),
        "avg_bitwidth": study.best_trial.user_attrs.get("avg_bitwidth", None),
        "runtime_average_wer": study.best_trial.user_attrs.get("runtime_average_wer", None),
        "runtime_average_latency": study.best_trial.user_attrs.get("runtime_average_latency", None),
        "runtime_inference_energy_consumption": study.best_trial.user_attrs.get("runtime_inference_energy_consumption", None),
    }
    
    # Save best configuration to JSON
    import json
    with open("best_configuration.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    logger.info("Best configuration saved to best_configuration.json")
    
    return study, results_df, baseline_model_data

if __name__ == "__main__":
    main()