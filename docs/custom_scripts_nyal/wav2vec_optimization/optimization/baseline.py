"""
Baseline metrics evaluation for Wav2Vec2 optimization.
"""

import logging
from chop.passes.graph import (
    runtime_analysis_pass,
    calculate_avg_bits_mg_analysis_pass,
)
import chop.passes as passes

from config import BATCH_SIZE

# Set up logging
logger = logging.getLogger(__name__)

def run_baseline_metrics(mg, data_module, checkpoint, dataset_name, decoder, tokenizer, ctc_head):
    """Run baseline metrics to establish reference performance"""
    logger.info("Running baseline metrics...")
    
    # Configure runtime analysis
    runtime_analysis_config = {
        "num_batches": 15,
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
    

    mg, _ = passes.init_metadata_analysis_pass(mg)

    mg, _ = passes.add_common_metadata_analysis_pass(mg)

    _, baseline_results = runtime_analysis_pass(mg, pass_args=runtime_analysis_config)
    
    # Run bit width analysis for baseline
    _, bitwidth_results = calculate_avg_bits_mg_analysis_pass(mg)
    baseline_results.update({"avg_bitwidth": bitwidth_results.get("average_bitwidth", 32)})
    
    logger.info("Baseline metrics complete")
    logger.info("Baseline Results:")
    for k, v in baseline_results.items():
        logger.info(f"  {k}: {v}")
    
    return baseline_results, mg
