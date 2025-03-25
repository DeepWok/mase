"""
SmoothQuant implementation for Wav2Vec2 optimization.
"""

import logging
from chop.passes.graph import onnx_runtime_interface_pass
from config import BATCH_SIZE

# Set up logging
logger = logging.getLogger(__name__)

def apply_smoothquant(mg, alpha, data_module, checkpoint, dataset_name):
    """Apply SmoothQuant to the model with specified alpha value"""
    logger.info(f"Applying SmoothQuant with alpha={alpha}")
    
    # SmoothQuant configuration
    smoothquant_config = {
        "smoothquant": True,
        "alpha": alpha,
        "model": checkpoint,
        "task": "ctc",
        "dataset": dataset_name,
        "accelerator": "cuda",
        "data_module": data_module,
        "batch_size": BATCH_SIZE,
    }
    
    # Run ONNX interface pass to apply SmoothQuant
    mg_smoothed, onnx_meta = onnx_runtime_interface_pass(mg, pass_args=smoothquant_config)
    
    return mg_smoothed, onnx_meta
