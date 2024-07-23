from .analysis import calculate_avg_bits_module_analysis_pass
from .transforms import quantize_module_transform_pass, resharding_transform_pass

ANALYSIS_PASSES = ["calculate_avg_bits_module_analysis_pass"]

TRANSFORM_PASSES = [
    "quantize_module_transform_pass",
]


PASSES = {
    # analysis
    "calculate_avg_bits": calculate_avg_bits_module_analysis_pass,
    # transform
    "quantize": quantize_module_transform_pass,
}
