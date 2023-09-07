import torch

# from .flop_estimator import run_flop_estimator
# from .statistical_profiler import run_statistical_profiler
from .add_metadata import (
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from .init_metadata import init_metadata_analysis_pass
from .report import (
    report_graph_analysis_pass,
    report_node_hardware_type_analysis_pass,
    report_node_meta_param_analysis_pass,
    report_node_shape_analysis_pass,
    report_node_type_analysis_pass,
)
from .statistical_profiler import profile_statistics_analysis_pass
from .verify import (
    verify_common_metadata_analysis_pass,
    verify_hardware_metadata_analysis_pass,
    verify_metadata_analysis_pass,
    verify_software_metadata_analysis_pass,
)
from .fpgaconvnet import fpgaconvnet_optimiser_analysis_pass
from .total_bits_estimator import (
    total_bits_mg_analysis_pass,
    total_bits_module_analysis_pass,
)
