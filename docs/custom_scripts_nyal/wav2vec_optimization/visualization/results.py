"""
Results processing for optimization study with comprehensive layer breakdown for all trials.
"""

import pandas as pd
import logging
import json
from config import ENHANCED_OBJECTIVE

# Set up logging
logger = logging.getLogger(__name__)

def process_study_results(study, enhanced=ENHANCED_OBJECTIVE):
    """Process study results into a comprehensive DataFrame with full layer breakdown for all trials"""
    logger.info("Processing study results...")
    
    # Store summary results (one row per trial)
    summary_results = []
    
    # Store detailed layer-wise results (one row per layer per trial)
    layer_results = []
    
    for t in sorted(study.trials, key=lambda t: t.number):
        # Basic trial info for summary
        row = {
            "trial_number": t.number,
            "composite_metric": t.value,
            "composite_score": t.user_attrs.get("composite_score", None),
            "new_composite_metric": t.user_attrs.get("new_composite_metric", None),
            "new_composite_score": t.user_attrs.get("new_composite_score", None),
        }
        
        # Add the normalized metrics
        row.update({
            "normalized_wer": t.user_attrs.get("normalized_wer", None),
            "normalized_latency": t.user_attrs.get("normalized_latency", None),
            "normalized_energy": t.user_attrs.get("normalized_energy", None),
            "normalized_bitwidth": t.user_attrs.get("normalized_bitwidth", None),
        })
        
        # Check if this is a mixed precision trial
        has_precision_decisions = "precision_decisions" in t.user_attrs
        row["is_mixed_precision"] = has_precision_decisions
        
        # Add precision decision counts if this is a mixed precision trial
        if has_precision_decisions:
            precision_decisions = t.user_attrs["precision_decisions"]
            
            # Count occurrences of each quantization type
            type_counts = {}
            for layer_type in precision_decisions.values():
                type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
            
            # Store counts in the row
            for type_name, count in type_counts.items():
                row[f"count_{type_name}"] = count
                
            # Calculate the percentage of each type
            total_layers = len(precision_decisions)
            for type_name, count in type_counts.items():
                row[f"percent_{type_name}"] = (count / total_layers) * 100 if total_layers > 0 else 0
                
            # Add detailed layer-wise data
            for layer_name, quant_type in precision_decisions.items():
                layer_row = {
                    "trial_number": t.number,
                    "layer_name": layer_name,
                    "quantization_type": quant_type,
                }
                
                # Add layer group information
                group = layer_name.split('.')[0] if '.' in layer_name else 'other'
                layer_row["layer_group"] = group
                
                # Add configuration parameters for this layer
                for param_name in t.params:
                    if param_name.startswith(f"{layer_name}_"):
                        # Extract the parameter type (e.g., weight_width, data_in_width)
                        param_type = param_name.replace(f"{layer_name}_", "")
                        layer_row[param_type] = t.params[param_name]
                
                # Add performance metrics to each layer for easier correlation analysis
                for metric_name in ["runtime_average_wer", "runtime_average_latency", 
                                   "runtime_inference_energy_consumption", "avg_bitwidth"]:
                    metric_value = t.user_attrs.get(metric_name.replace("runtime_", ""), None)
                    if metric_value is None:
                        metric_value = t.user_attrs.get(metric_name, None)
                    layer_row[metric_name] = metric_value
                
                layer_results.append(layer_row)
        else:
            # For non-mixed precision trials, record the single quantization method
            row["quantization_method"] = t.user_attrs.get("quantization_method", "N/A")
        
        # Trial parameters
        row.update({
            "pruning_method": t.params.get("pruning_method", "N/A"),
            "smoothquant_alpha": t.params.get("smoothquant_alpha", None),
        })
        
        # Pruning parameters
        row.update({
            "pruning_sparsity": t.params.get("pruning_sparsity", None),
            "structured_sparsity": t.params.get("structured_sparsity", None),
            "overall_sparsity": t.user_attrs.get("overall_sparsity", None),
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
        row["data_avg_bitwidth"] = t.user_attrs.get("data_avg_bitwidth", None)
        row["weight_avg_bitwidth"] = t.user_attrs.get("weight_avg_bitwidth", None)
        row["bitwidth_reduction"] = t.user_attrs.get("bitwidth_reduction", None)
        
        summary_results.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_results)
    
    # Save summary results to CSV
    summary_csv_name = "mixed_precision_summary_results.csv"
    summary_df.to_csv(summary_csv_name, index=False)
    logger.info(f"Summary results saved to {summary_csv_name}")
    
    # Create comprehensive layer-wise DataFrame
    if layer_results:
        layer_df = pd.DataFrame(layer_results)
        
        # Save comprehensive layer results to CSV
        layer_csv_name = "all_trials_layer_precision.csv"
        layer_df.to_csv(layer_csv_name, index=False)
        logger.info(f"Comprehensive layer-wise results for all trials saved to {layer_csv_name}")
        
        # Create pivot tables for easier analysis
        try:
            # Create a pivot table showing quantization type by layer name across trials
            quant_pivot = pd.pivot_table(
                layer_df, 
                values='trial_number',
                index=['layer_name'], 
                columns=['quantization_type'], 
                aggfunc='count', 
                fill_value=0
            )
            quant_pivot.to_csv("layer_quantization_distribution.csv")
            logger.info("Layer quantization distribution saved to layer_quantization_distribution.csv")
            
            # Create a pivot table showing the average bit width by layer across trials
            if 'weight_width' in layer_df.columns:
                bitwidth_pivot = pd.pivot_table(
                    layer_df, 
                    values='weight_width',
                    index=['layer_name'], 
                    columns=['quantization_type'], 
                    aggfunc='mean', 
                    fill_value=0
                )
                bitwidth_pivot.to_csv("layer_bitwidth_distribution.csv")
                logger.info("Layer bit width distribution saved to layer_bitwidth_distribution.csv")
                
            # Create a pivot table showing quantization choices for best performing trials
            if 'runtime_average_wer' in layer_df.columns:
                # Get the top 5 trial numbers by WER
                top_trials = summary_df.nsmallest(5, 'runtime_average_wer')['trial_number'].tolist()
                
                # Filter layer_df to only include these trials
                top_layer_df = layer_df[layer_df['trial_number'].isin(top_trials)]
                
                # Create pivot table
                top_quant_pivot = pd.pivot_table(
                    top_layer_df, 
                    values='trial_number',
                    index=['layer_name'], 
                    columns=['quantization_type'], 
                    aggfunc='count', 
                    fill_value=0
                )
                top_quant_pivot.to_csv("top_trials_layer_quantization.csv")
                logger.info("Top trials layer quantization saved to top_trials_layer_quantization.csv")
                
        except Exception as e:
            logger.warning(f"Error creating pivot tables: {e}")
    
    return summary_df, layer_df if layer_results else None

def generate_mixed_precision_analysis(study_results_df):
    """Generate additional analysis for mixed precision results"""
    logger.info("Generating mixed precision analysis...")
    
    # Only consider mixed precision trials (trials with precision decisions)
    mp_df = study_results_df[study_results_df["is_mixed_precision"] == True]
    
    if mp_df.empty:
        logger.warning("No mixed precision trials found in the results")
        return
    
    # 1. Find the most common quantization type for each metric
    metrics = ["runtime_average_wer", "runtime_average_latency", "runtime_inference_energy_consumption"]
    quant_types = [col.replace("percent_", "") for col in mp_df.columns if col.startswith("percent_")]
    
    best_types = {}
    for metric in metrics:
        if metric in mp_df.columns:
            # For WER, lower is better
            direction = "min" if "wer" in metric else "min"
            
            # Find the top 5 trials for this metric
            if direction == "min":
                top_trials = mp_df.nsmallest(5, metric)
            else:
                top_trials = mp_df.nlargest(5, metric)
            
            # Calculate the average percentage for each quantization type
            type_percentages = {}
            for qtype in quant_types:
                percent_col = f"percent_{qtype}"
                if percent_col in top_trials.columns:
                    type_percentages[qtype] = top_trials[percent_col].mean()
            
            # Store the results
            best_types[metric] = type_percentages
    
    # Save the analysis
    with open("mixed_precision_type_analysis.json", "w") as f:
        json.dump(best_types, f, indent=2)
    
    logger.info("Mixed precision analysis saved to mixed_precision_type_analysis.json")
    
    # 2. Create a correlation matrix between layer patterns and performance
    if len(quant_types) > 1:
        correlation_data = {}
        for metric in metrics:
            if metric in mp_df.columns:
                correlations = {}
                for qtype in quant_types:
                    percent_col = f"percent_{qtype}"
                    if percent_col in mp_df.columns:
                        corr = mp_df[percent_col].corr(mp_df[metric])
                        correlations[qtype] = corr
                correlation_data[metric] = correlations
        
        # Save the correlations
        with open("precision_type_correlations.json", "w") as f:
            json.dump(correlation_data, f, indent=2)
        
        logger.info("Precision type correlations saved to precision_type_correlations.json")
    
    return best_types