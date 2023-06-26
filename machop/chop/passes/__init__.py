from .analysis import init_metadata_analysis_pass, add_common_metadata_analysis_pass
from .transforms import quantize_transform_pass

analysis_passes = ["init_metadata", "add_common_metadata"]
transform_passes = ["quantize"]

passes = {
    "init_metadata": init_metadata_analysis_pass,
    "add_common_metadata": add_common_metadata_analysis_pass,
    "quantize": quantize_transform_pass,
}