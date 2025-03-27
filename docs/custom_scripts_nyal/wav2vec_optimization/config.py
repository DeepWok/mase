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
EPOCHS = 1
ENHANCED_OBJECTIVE = False  # Whether to use phase-by-phase training and evaluation
CHECKPOINT = "facebook/wav2vec2-base-960h"
TOKENIZER_CHECKPOINT = "facebook/wav2vec2-base-960h"
DATASET_NAME = "nyalpatel/condensed_librispeech_asr"
CREATE_VISUALISATONS = False

# Define global constants for parameter choices to ensure consistency
# These are used by the quantization module to ensure parameter consistency
WEIGHT_WIDTH_CHOICES = [8, 16, 32]
WEIGHT_FRAC_WIDTH_CHOICES = [4, 8, 16]
DATA_IN_WIDTH_CHOICES = [8, 16, 32]
DATA_IN_FRAC_WIDTH_CHOICES = [4, 8, 16]
BIAS_WIDTH_CHOICES = [8, 16, 32]
BIAS_FRAC_WIDTH_CHOICES = [4, 8, 16]

# Minifloat constants
WEIGHT_EXPONENT_WIDTH_CHOICES = [3, 5, 8]
WEIGHT_EXPONENT_BIAS_CHOICES = [7, 15, 31]  # Include all possible values across all types
DATA_IN_EXPONENT_WIDTH_CHOICES = [3, 5, 8]
DATA_IN_EXPONENT_BIAS_CHOICES = [7, 15, 31]  # Include all possible values across all types
BIAS_EXPONENT_WIDTH_CHOICES = [3, 5, 8]
BIAS_EXPONENT_BIAS_CHOICES = [7, 15, 31]  # Include all possible values across all types

# Special handling for LinearLog, making sure to use the same choices as other types
LOG_EXPONENT_BIAS_CHOICES = [0, 7, 15, 31]  # Ensuring all possible values are included


def define_search_space():
    """Define the global search space for all optimization phases"""
    logger.info("Defining global search space...")
    
    # Search space for quantization
    quantization_methods = [
        ("full_precision", nn.Linear),  # Baseline for comparison
        ("integer", LinearInteger),     # INT quantization
        ("minifloat_denorm", LinearMinifloatDenorm),  # Minifloat with denormalized numbers
        ("minifloat_ieee", LinearMinifloatIEEE),      # IEEE-style minifloat
        ("log", LinearLog),             # Logarithmic quantization
        ("block_fp", LinearBlockFP),    # Block floating point quantization
    ]
    
    # Bit width configurations for precision
    bit_width_configs = {
        "weight_width": WEIGHT_WIDTH_CHOICES,
        "weight_frac_width": WEIGHT_FRAC_WIDTH_CHOICES,
        "data_in_width": DATA_IN_WIDTH_CHOICES,
        "data_in_frac_width": DATA_IN_FRAC_WIDTH_CHOICES,
        "bias_width": BIAS_WIDTH_CHOICES,
        "bias_frac_width": BIAS_FRAC_WIDTH_CHOICES,
    }
    
    # Minifloat-specific configs
    minifloat_configs = {
        "weight_exponent_width": WEIGHT_EXPONENT_WIDTH_CHOICES,
        "weight_exponent_bias": WEIGHT_EXPONENT_BIAS_CHOICES,
        "data_in_exponent_width": DATA_IN_EXPONENT_WIDTH_CHOICES,
        "data_in_exponent_bias": DATA_IN_EXPONENT_BIAS_CHOICES,
        "bias_exponent_width": BIAS_EXPONENT_WIDTH_CHOICES, 
        "bias_exponent_bias": BIAS_EXPONENT_BIAS_CHOICES,
    }
    
    # Log-specific configs
    log_configs = {
        "weight_exponent_bias": LOG_EXPONENT_BIAS_CHOICES,
        "data_in_exponent_bias": LOG_EXPONENT_BIAS_CHOICES,
        "bias_exponent_bias": LOG_EXPONENT_BIAS_CHOICES,
    }
    
    # Search space for pruning
    pruning_methods = ["hwpq", "random", "l1-norm", "movement", "snip"]
    pruning_sparsity_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    structured_sparsity_options = [True, False]  # True for structured pruning patterns
    
    # Search space for SmoothQuant
    smoothquant_alpha_values = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Mixed precision configuration
    mixed_precision = {
        "quantization_classes": [
            nn.Linear,  # Full precision
            LinearInteger,
            LinearMinifloatDenorm,
            LinearMinifloatIEEE,
            LinearLog,
            LinearBlockFP,
        ],
        "width_choices": WEIGHT_WIDTH_CHOICES,
        "frac_width_choices": WEIGHT_FRAC_WIDTH_CHOICES,
        "exponent_width_choices": WEIGHT_EXPONENT_WIDTH_CHOICES,
        "exponent_bias_choices": WEIGHT_EXPONENT_BIAS_CHOICES,
    }
    
    search_space = {
        "quantization": {
            "methods": quantization_methods,
            "bit_width_configs": bit_width_configs,
            "minifloat_configs": minifloat_configs,
            "log_configs": log_configs,
        },
        "pruning": {
            "methods": pruning_methods,
            "sparsity_levels": pruning_sparsity_levels,
            "structured_options": structured_sparsity_options,
        },
        "smoothquant": {
            "alpha_values": smoothquant_alpha_values,
        },
        "mixed_precision": mixed_precision,
    }
    
    logger.info("Global search space defined:\n%s", pprint.pformat(search_space))
    return search_space