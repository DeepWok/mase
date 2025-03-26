"""
Global configuration and search space definition for Wav2Vec2 optimization.
"""

import torch.nn as nn
import pprint
import logging
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configuration
BATCH_SIZE = 4
NUM_TRIALS = 30
EPOCHS = 0.1
ENHANCED_OBJECTIVE = False  # Whether to use phase-by-phase training and evaluation
CHECKPOINT = "facebook/wav2vec2-base-960h"
TOKENIZER_CHECKPOINT = "facebook/wav2vec2-base-960h"
DATASET_NAME = "nyalpatel/condensed_librispeech_asr"


def define_search_space():
    """Define the global search space for all optimization phases"""
    logger.info("Defining global search space...")
    
    # Search space for quantization
    quantization_methods = [
        ("full_precision", nn.Linear),  # Baseline for comparison
        ("integer", LinearInteger),     # INT quantization
        ("minifloat_denorm", LinearMinifloatDenorm),  # Minifloat with denormalized numbers
        ("minifloat_ieee", LinearMinifloatIEEE),      # IEEE-style minifloat         # Logarithmic quantization
    ]
    
    # Bit width configurations for precision
    bit_width_configs = {
        "weight_width": [8, 16, 32],
        "weight_frac_width": [4, 8, 16],
        "data_in_width": [8, 16, 32],
        "data_in_frac_width": [4, 8, 16],
        "bias_width": [8, 16, 32],
        "bias_frac_width": [4, 8, 16],
    }
    
    # Minifloat-specific configs
    minifloat_configs = {
        "weight_exponent_width": [3, 5, 8],
        "weight_exponent_bias": [7, 15, 31],
        "data_in_exponent_width": [3, 5, 8],
        "data_in_exponent_bias": [7, 15, 31],
        "bias_exponent_width": [3, 5, 8], 
        "bias_exponent_bias": [7, 15, 31],
    }
    
    # Search space for pruning
    pruning_methods = ["hwpq", "random", "l1-norm", "movement", "snip"]
    pruning_sparsity_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    structured_sparsity_options = [True, False]  # True for structured pruning patterns
    
    # Search space for SmoothQuant
    smoothquant_alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    search_space = {
        "quantization": {
            "methods": quantization_methods,
            "bit_width_configs": bit_width_configs,
            "minifloat_configs": minifloat_configs,
        },
        "pruning": {
            "methods": pruning_methods,
            "sparsity_levels": pruning_sparsity_levels,
            "structured_options": structured_sparsity_options,
        },
        "smoothquant": {
            "alpha_values": smoothquant_alpha_values,
        }
    }
    
    logger.info("Global search space defined:\n%s", pprint.pformat(search_space))
    return search_space
