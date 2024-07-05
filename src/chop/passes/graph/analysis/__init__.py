import torch

# from .flop_estimator import run_flop_estimator
# from .statistical_profiler import run_statistical_profiler
from .add_metadata import (
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from .init_metadata import init_metadata_analysis_pass
from .hardware import run_cosim_analysis_pass, get_synthesis_results
from .report import (
    report_graph_analysis_pass,
    report_node_hardware_type_analysis_pass,
    report_node_meta_param_analysis_pass,
    report_node_shape_analysis_pass,
    report_node_type_analysis_pass,
)
from .statistical_profiler import profile_statistics_analysis_pass
from .verilog import test_verilog_analysis_pass
from .verify import (
    verify_common_metadata_analysis_pass,
    verify_hardware_metadata_analysis_pass,
    verify_metadata_analysis_pass,
    verify_software_metadata_analysis_pass,
)
from .quantization import calculate_avg_bits_mg_analysis_pass

from .pruning import (
    add_pruning_metadata_analysis_pass,
    add_natural_sparsity_metadata_analysis_pass,
    hook_inspection_analysis_pass,
)
