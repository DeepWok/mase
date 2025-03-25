"""
Optimization package for Wav2Vec2 model.
"""

from optimization.baseline import run_baseline_metrics
from optimization.pruning import apply_pruning, calculate_pruning_metrics
from optimization.smoothquant import apply_smoothquant
from optimization.quantization import apply_quantization

__all__ = [
    'run_baseline_metrics',
    'apply_pruning',
    'calculate_pruning_metrics',
    'apply_smoothquant',
    'apply_quantization',
]