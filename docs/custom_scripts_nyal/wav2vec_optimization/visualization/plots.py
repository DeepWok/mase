"""
Visualization creation for optimization study.
"""

import os
import matplotlib.pyplot as plt
import logging
import optuna
from config import ENHANCED_OBJECTIVE

# Set up logging
logger = logging.getLogger(__name__)

def create_visualizations(study, results_df, baseline_metrics, enhanced=ENHANCED_OBJECTIVE):
    """Create visualizations for the optimization study"""
    logger.info("Creating visualizations...")
    
    # Set up directories for saving plots
    os.makedirs("plots", exist_ok=True)
    
    # 1. Optimization history plot
    plt.figure(figsize=(12, 6))
    plt.title("Optimization History")
    plt.plot(results_df["trial_number"], results_df["composite_metric"], "o-")
    plt.xlabel("Trial Number")
    plt.ylabel("Composite Metric (higher is better)")
    plt.grid(True)
    plt.savefig("plots/optimization_history.png")
    plt.close()
    
    # 2. Parallel coordinate plot for hyperparameters
    try:
        fig = optuna.visualization.plot_parallel_coordinate(
            study, 
            params=["pruning_method", "pruning_sparsity", "smoothquant_alpha", "quantization_method_idx"]
        )
        fig.write_image("plots/parallel_coordinate.png")
    except Exception as e:
        logger.warning(f"Could not create parallel coordinate plot: {e}")
    
    # 3. Parameter importance
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("plots/param_importances.png")
    except Exception as e:
        logger.warning(f"Could not create parameter importance plot: {e}")
    
    # 4. WER vs Latency scatter plot - final model performance
    wer_column = "final_average_wer" if enhanced else "runtime_average_wer"
    latency_column = "final_average_latency" if enhanced else "runtime_average_latency"
    
    if wer_column in results_df.columns and latency_column in results_df.columns:
        plt.figure(figsize=(10, 8))
        
        # Color points by quantization method
        quant_methods = results_df["quantization_method"].unique()
        cm = plt.cm.get_cmap("tab10", len(quant_methods))
        
        for i, method in enumerate(quant_methods):
            method_df = results_df[results_df["quantization_method"] == method]
            plt.scatter(
                method_df[latency_column], 
                method_df[wer_column],
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
        plt.savefig("plots/wer_vs_latency.png")
        plt.close()
    
    # 5. Sparsity vs performance metrics
    if "overall_sparsity" in results_df.columns:
        sparsity_df = results_df[results_df["overall_sparsity"].notnull()]
        
        if not sparsity_df.empty:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # WER vs Sparsity
            wer_col = "final_average_wer" if enhanced else "runtime_average_wer"
            if wer_col in sparsity_df.columns:
                axs[0].scatter(sparsity_df["overall_sparsity"], sparsity_df[wer_col])
                axs[0].set_xlabel("Sparsity")
                axs[0].set_ylabel("WER (%)")
                axs[0].set_title("WER vs Sparsity")
                axs[0].grid(True)
            
            # Latency vs Sparsity
            latency_col = "final_average_latency" if enhanced else "runtime_average_latency"
            if latency_col in sparsity_df.columns:
                axs[1].scatter(sparsity_df["overall_sparsity"], sparsity_df[latency_col])
                axs[1].set_xlabel("Sparsity")
                axs[1].set_ylabel("Latency (ms)")
                axs[1].set_title("Latency vs Sparsity")
                axs[1].grid(True)
            
            # Energy vs Sparsity
            energy_col = "final_inference_energy_consumption" if enhanced else "runtime_inference_energy_consumption"
            if energy_col in sparsity_df.columns:
                axs[2].scatter(sparsity_df["overall_sparsity"], sparsity_df[energy_col])
                axs[2].set_xlabel("Sparsity")
                axs[2].set_ylabel("Energy Consumption")
                axs[2].set_title("Energy vs Sparsity")
                axs[2].grid(True)
            
            plt.tight_layout()
            plt.savefig("plots/sparsity_performance.png")
            plt.close()
    
    # 6. Bit width vs performance metrics
    if "avg_bitwidth" in results_df.columns:
        bitwidth_df = results_df[results_df["avg_bitwidth"].notnull()]
        
        if not bitwidth_df.empty:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # WER vs Bit Width
            wer_col = "final_average_wer" if enhanced else "runtime_average_wer"
            if wer_col in bitwidth_df.columns:
                axs[0].scatter(bitwidth_df["avg_bitwidth"], bitwidth_df[wer_col])
                axs[0].set_xlabel("Average Bit Width")
                axs[0].set_ylabel("WER (%)")
                axs[0].set_title("WER vs Bit Width")
                axs[0].grid(True)
            
            # Latency vs Bit Width
            latency_col = "final_average_latency" if enhanced else "runtime_average_latency"
            if latency_col in bitwidth_df.columns:
                axs[1].scatter(bitwidth_df["avg_bitwidth"], bitwidth_df[latency_col])
                axs[1].set_xlabel("Average Bit Width")
                axs[1].set_ylabel("Latency (ms)")
                axs[1].set_title("Latency vs Bit Width")
                axs[1].grid(True)
            
            # Energy vs Bit Width
            energy_col = "final_inference_energy_consumption" if enhanced else "runtime_inference_energy_consumption"
            if energy_col in bitwidth_df.columns:
                axs[2].scatter(bitwidth_df["avg_bitwidth"], bitwidth_df[energy_col])
                axs[2].set_xlabel("Average Bit Width")
                axs[2].set_ylabel("Energy Consumption")
                axs[2].set_title("Energy vs Bit Width")
                axs[2].grid(True)
            
            plt.tight_layout()
            plt.savefig("plots/bitwidth_performance.png")
            plt.close()
    
    # 7. Quantization method comparison box plots
    plt.figure(figsize=(15, 10))
    
    # Compare WER across methods
    plt.subplot(2, 2, 1)
    wer_col = "final_average_wer" if enhanced else "runtime_average_wer"
    if "quantization_method" in results_df.columns and wer_col in results_df.columns:
        results_df.boxplot(column=wer_col, by="quantization_method", ax=plt.gca())
        plt.ylabel("WER (%)")
        plt.title("WER by Quantization Method")
        plt.suptitle("")  # Remove pandas-generated suptitle
        plt.xticks(rotation=45)
    
    # Compare Latency across methods
    plt.subplot(2, 2, 2)
    latency_col = "final_average_latency" if enhanced else "runtime_average_latency"
    if "quantization_method" in results_df.columns and latency_col in results_df.columns:
        results_df.boxplot(column=latency_col, by="quantization_method", ax=plt.gca())
        plt.ylabel("Latency (ms)")
        plt.title("Latency by Quantization Method")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare Energy across methods
    plt.subplot(2, 2, 3)
    energy_col = "final_inference_energy_consumption" if enhanced else "runtime_inference_energy_consumption"
    if "quantization_method" in results_df.columns and energy_col in results_df.columns:
        results_df.boxplot(column=energy_col, by="quantization_method", ax=plt.gca())
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
    plt.savefig("plots/method_comparison.png")
    plt.close()
    
    # 8. For enhanced trials, create phase comparison plots
    if enhanced:
        # Performance across phases
        if all(col in results_df.columns for col in ["pruned_average_wer", "smoothed_average_wer", "final_average_wer"]):
            plt.figure(figsize=(12, 8))
            
            # Select top N trials by final performance
            top_n = min(10, len(results_df))
            top_trials = results_df.nsmallest(top_n, "final_average_wer")
            
            # Extract phase data
            phases = ["Baseline", "After Pruning", "After SmoothQuant", "After Quantization"]
            trial_data = {}
            
            for _, row in top_trials.iterrows():
                trial_num = row["trial_number"]
                trial_data[trial_num] = [
                    baseline_metrics.get("Average WER", 0),
                    row.get("pruned_average_wer", 0),
                    row.get("smoothed_average_wer", 0),
                    row.get("final_average_wer", 0)
                ]
            
            # Plot
            for trial_num, values in trial_data.items():
                plt.plot(phases, values, marker='o', label=f"Trial {trial_num}")
            
            plt.xlabel("Optimization Phase")
            plt.ylabel("WER (%)")
            plt.title("WER Progression Through Optimization Phases")
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("plots/phase_comparison_wer.png")
            plt.close()
            
            # Similar plot for latency
            if all(col in results_df.columns for col in ["pruned_average_latency", "smoothed_average_latency", "final_average_latency"]):
                plt.figure(figsize=(12, 8))
                
                # Select top N trials by final latency
                top_latency_trials = results_df.nsmallest(top_n, "final_average_latency")
                
                # Extract phase data
                latency_data = {}
                
                for _, row in top_latency_trials.iterrows():
                    trial_num = row["trial_number"]
                    latency_data[trial_num] = [
                        baseline_metrics.get("Average Latency", 0),
                        row.get("pruned_average_latency", 0),
                        row.get("smoothed_average_latency", 0),
                        row.get("final_average_latency", 0)
                    ]
                
                # Plot
                for trial_num, values in latency_data.items():
                    plt.plot(phases, values, marker='o', label=f"Trial {trial_num}")
                
                plt.xlabel("Optimization Phase")
                plt.ylabel("Latency (ms)")
                plt.title("Latency Progression Through Optimization Phases")
                plt.grid(True)
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig("plots/phase_comparison_latency.png")
                plt.close()
    
    # 9. Pruning method comparison box plots
    plt.figure(figsize=(15, 10))
    
    # Compare WER across pruning methods
    plt.subplot(2, 2, 1)
    wer_col = "final_average_wer" if enhanced else "runtime_average_wer"
    if "pruning_method" in results_df.columns and wer_col in results_df.columns:
        results_df.boxplot(column=wer_col, by="pruning_method", ax=plt.gca())
        plt.ylabel("WER (%)")
        plt.title("WER by Pruning Method")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare Latency across pruning methods
    plt.subplot(2, 2, 2)
    latency_col = "final_average_latency" if enhanced else "runtime_average_latency"
    if "pruning_method" in results_df.columns and latency_col in results_df.columns:
        results_df.boxplot(column=latency_col, by="pruning_method", ax=plt.gca())
        plt.ylabel("Latency (ms)")
        plt.title("Latency by Pruning Method")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare sparsity across pruning methods
    plt.subplot(2, 2, 3)
    if "pruning_method" in results_df.columns and "overall_sparsity" in results_df.columns:
        results_df.boxplot(column="overall_sparsity", by="pruning_method", ax=plt.gca())
        plt.ylabel("Overall Sparsity")
        plt.title("Sparsity by Pruning Method")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare energy across pruning methods
    plt.subplot(2, 2, 4)
    energy_col = "final_inference_energy_consumption" if enhanced else "runtime_inference_energy_consumption"
    if "pruning_method" in results_df.columns and energy_col in results_df.columns:
        results_df.boxplot(column=energy_col, by="pruning_method", ax=plt.gca())
        plt.ylabel("Energy Consumption")
        plt.title("Energy by Pruning Method")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("plots/pruning_method_comparison.png")
    plt.close()
    
    # 10. SmoothQuant alpha comparison
    plt.figure(figsize=(15, 10))
    
    # Compare WER across alpha values
    plt.subplot(2, 2, 1)
    wer_col = "final_average_wer" if enhanced else "runtime_average_wer"
    if "smoothquant_alpha" in results_df.columns and wer_col in results_df.columns:
        results_df.boxplot(column=wer_col, by="smoothquant_alpha", ax=plt.gca())
        plt.ylabel("WER (%)")
        plt.title("WER by SmoothQuant Alpha")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare Latency across alpha values
    plt.subplot(2, 2, 2)
    latency_col = "final_average_latency" if enhanced else "runtime_average_latency"
    if "smoothquant_alpha" in results_df.columns and latency_col in results_df.columns:
        results_df.boxplot(column=latency_col, by="smoothquant_alpha", ax=plt.gca())
        plt.ylabel("Latency (ms)")
        plt.title("Latency by SmoothQuant Alpha")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare bit width across alpha values
    plt.subplot(2, 2, 3)
    if "smoothquant_alpha" in results_df.columns and "avg_bitwidth" in results_df.columns:
        results_df.boxplot(column="avg_bitwidth", by="smoothquant_alpha", ax=plt.gca())
        plt.ylabel("Average Bit Width")
        plt.title("Bit Width by SmoothQuant Alpha")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    # Compare energy across alpha values
    plt.subplot(2, 2, 4)
    energy_col = "final_inference_energy_consumption" if enhanced else "runtime_inference_energy_consumption"
    if "smoothquant_alpha" in results_df.columns and energy_col in results_df.columns:
        results_df.boxplot(column=energy_col, by="smoothquant_alpha", ax=plt.gca())
        plt.ylabel("Energy Consumption")
        plt.title("Energy by SmoothQuant Alpha")
        plt.suptitle("")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("plots/smoothquant_alpha_comparison.png")
    plt.close()
    
    logger.info("Visualizations created and saved to 'plots/' directory")
