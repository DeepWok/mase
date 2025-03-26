"""
Results processing for optimization study.
"""

import pandas as pd
import logging
from config import ENHANCED_OBJECTIVE

# Set up logging
logger = logging.getLogger(__name__)

def process_study_results(study, enhanced=ENHANCED_OBJECTIVE):
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
            "pruning_method": t.params.get("pruning_method", "N/A"),
            "smoothquant_alpha": t.params.get("smoothquant_alpha", None),
        })
        
        # Pruning parameters
        row.update({
            "pruning_sparsity": t.params.get("pruning_sparsity", None),
            "structured_sparsity": t.params.get("structured_sparsity", None),
            "overall_sparsity": t.user_attrs.get("pruning_overall_sparsity", None),
        })
        
        # Performance metrics for final model
        runtime_metrics = ["average_wer", "average_latency", "average_rtf", 
                          "average_gpu_power_usage", "inference_energy_consumption"]
        
        # Add metrics for the phases if using enhanced objective
        if enhanced:
            # Pruning phase metrics
            for metric in runtime_metrics:
                row[f"pruned_{metric}"] = t.user_attrs.get(f"pruned_{metric}", None)
            
            # SmoothQuant phase metrics
            for metric in runtime_metrics:
                row[f"smoothed_{metric}"] = t.user_attrs.get(f"smoothed_{metric}", None)
            
            # Final phase metrics
            for metric in runtime_metrics:
                row[f"final_{metric}"] = t.user_attrs.get(f"final_{metric}", None)
                row[f"pct_change_{metric}"] = t.user_attrs.get(f"pct_change_{metric}", None)
            
            # Training metrics for each phase
            row["pruned_eval_wer"] = t.user_attrs.get("pruned_eval_wer", None)
            row["pruned_eval_loss"] = t.user_attrs.get("pruned_eval_loss", None)
            row["smoothed_eval_wer"] = t.user_attrs.get("smoothed_eval_wer", None)
            row["smoothed_eval_loss"] = t.user_attrs.get("smoothed_eval_loss", None)
            row["final_eval_wer"] = t.user_attrs.get("final_eval_wer", None)
            row["final_eval_loss"] = t.user_attrs.get("final_eval_loss", None)
        else:
            # Standard metrics for single-phase optimization
            for metric in runtime_metrics:
                row[f"runtime_{metric}"] = t.user_attrs.get(f"runtime_{metric}", None)
                row[f"pct_change_{metric}"] = t.user_attrs.get(f"pct_change_{metric}", None)
            
            # Add pytorch evaluation metrics
            row["eval_wer_pt"] = t.user_attrs.get("eval_wer_pt", None)
            row["eval_loss_pt"] = t.user_attrs.get("eval_loss_pt", None)
        
        # Bit width metrics
        row["avg_bitwidth"] = t.user_attrs.get("avg_bitwidth", None)
        row["bitwidth_reduction"] = t.user_attrs.get("bitwidth_reduction", None)
        
        # Store quantization parameters
        if t.user_attrs.get("quantization_method", "full_precision") != "full_precision":
            row["weight_width"] = t.params.get("weight_width", None)
            row["data_in_width"] = t.params.get("data_in_width", None)
            
            # Method-specific parameters
            if t.user_attrs.get("quantization_method", "") == "integer":
                row["weight_frac_width"] = t.params.get("weight_frac_width", None)
                row["data_in_frac_width"] = t.params.get("data_in_frac_width", None)
                row["bias_width"] = t.params.get("bias_width", None)
                row["bias_frac_width"] = t.params.get("bias_frac_width", None)
            elif t.user_attrs.get("quantization_method", "") in ["minifloat_denorm", "minifloat_ieee"]:
                row["weight_exponent_width"] = t.params.get("weight_exponent_width", None)
                row["weight_exponent_bias"] = t.params.get("weight_exponent_bias", None)
                row["data_in_exponent_width"] = t.params.get("data_in_exponent_width", None)
                row["data_in_exponent_bias"] = t.params.get("data_in_exponent_bias", None)
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    print(df)
    
    # Save results to CSV
    csv_name = "optuna_study_results.csv"
    df.to_csv(csv_name, index=False)
    logger.info(f"Study results saved to {csv_name}")
    
    return df
