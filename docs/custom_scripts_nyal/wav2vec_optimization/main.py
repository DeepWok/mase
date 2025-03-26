"""
Main script to run the Wav2Vec2 optimization.
"""

import logging
import torch
import argparse

import config
from config import define_search_space, NUM_TRIALS, CREATE_VISUALISATONS
from data_utils import import_model_and_dataset
from model_utils import setup_mase_graph, create_combined_model
from optimization.baseline import run_baseline_metrics
from search.study import run_optimization_study
from visualization.results import process_study_results
from visualization.plots import create_visualizations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    """Main function to run the optimization pipeline"""
    logger.info("Starting optimization pipeline")

    # Log the entire configuration from the config module (filter out built-ins)
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    logger.info("Entire config: %s", config_dict)
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. This will run very slowly on CPU.")
    else:
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    
    # 1. Define global search space
    search_space = define_search_space()
    logger.info("Global search space defined")
    
    # 2. Import model and dataset
    model_data = import_model_and_dataset()
    logger.info("Model and dataset imported")
    
    # 3. Setup MASE graph
    mg, dummy_in = setup_mase_graph(model_data["encoder"])
    model_data["mg"] = mg
    logger.info("MASE graph setup complete")
    
    # 4. Create combined model
    combined_model = create_combined_model(mg.model, model_data["ctc_head"], model_data["decoder"])
    logger.info("Combined model created")
    
    # 5. Run baseline metrics
    baseline_metrics, updated_mg = run_baseline_metrics(
        mg, 
        model_data["data_module"], 
        model_data["checkpoint"], 
        model_data["dataset_name"], 
        model_data["decoder"], 
        model_data["tokenizer"],
        model_data["ctc_head"]
    )
    model_data["mg"] = updated_mg
    logger.info("Baseline metrics collected")
    
    # Prepare baseline_model_data dictionary for optimization
    baseline_model_data = {
        **model_data,
        "baseline_metrics": baseline_metrics,
        "search_space": search_space,
    }
    
    # 6. Run optimization study
    logger.info("Starting optimization study")
    study = run_optimization_study(
        baseline_model_data, 
        n_trials=args.n_trials
    )
    
    # 7. Process results and create visualizations
    results_df = process_study_results(study)
    
    if CREATE_VISUALISATONS:
        create_visualizations(study, results_df, baseline_metrics)
    
    # 8. Print summary
    logger.info("Optimization pipeline complete")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best composite metric: {study.best_trial.value}")
    
    # Create a summary of the best configuration
    best_config = {
        "quantization_method": study.best_trial.user_attrs.get("quantization_method", "N/A"),
        "pruning_method": study.best_trial.params.get("pruning_method", "N/A"),
        "pruning_sparsity": study.best_trial.params.get("pruning_sparsity", None),
        "structured_sparsity": study.best_trial.params.get("structured_sparsity", None),
        "smoothquant_alpha": study.best_trial.params.get("smoothquant_alpha", None),
        "runtime_average_wer": study.best_trial.user_attrs.get("runtime_average_wer", None),
    }
    
    logger.info("Best configuration:")
    for k, v in best_config.items():
        logger.info(f"  {k}: {v}")
    
    return study, results_df, baseline_model_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wav2Vec2 Optimization Pipeline")
    parser.add_argument("--n_trials", type=int, default=NUM_TRIALS, 
                        help=f"Number of optimization trials (default: {NUM_TRIALS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)