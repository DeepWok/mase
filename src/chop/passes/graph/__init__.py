from .analysis import (
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
    profile_statistics_analysis_pass,
    report_graph_analysis_pass,
    report_node_hardware_type_analysis_pass,
    report_node_meta_param_analysis_pass,
    report_node_shape_analysis_pass,
    report_node_type_analysis_pass,
    verify_common_metadata_analysis_pass,
    run_cosim_analysis_pass,
    get_synthesis_results,
    calculate_avg_bits_mg_analysis_pass,
    add_pruning_metadata_analysis_pass,
    add_natural_sparsity_metadata_analysis_pass,
    hook_inspection_analysis_pass,
    runtime_analysis_pass,
)
from .transforms import (
    prune_transform_pass,
    prune_detach_hook_transform_pass,
    # prune_unwrap_transform_pass,
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
    conv_bn_fusion_transform_pass,
    logicnets_fusion_transform_pass,
    onnx_annotate_transform_pass,
    partition_to_multi_device_transform_pass,
    raise_granularity_transform_pass,
)

from .interface import (
    load_mase_graph_interface_pass,
    save_mase_graph_interface_pass,
    save_node_meta_param_interface_pass,
    load_node_meta_param_interface_pass,
    tensorrt_engine_interface_pass,
    onnx_runtime_interface_pass,
)

from .transforms.quantize.quant_parsers import parse_node_config
from .transforms.tensorrt import (
    tensorrt_calibrate_transform_pass,
    tensorrt_fine_tune_transform_pass,
    tensorrt_fake_quantize_transform_pass,
)

ANALYSIS_PASSES = [
    "init_metadata",
    "add_common_metadata",
    "add_hardware_metadata",
    "add_software_metadata",
    "profile_statistics",
    "report_graph",
    "report_node_hardware_type",
    "report_node_meta_param",
    "report_node_shape",
    "report_node_type",
    "total_bits_mg_analysis_pass",
    "run_cosim_analysis_pass",
    "get_synthesis_results",
    "calculate_avg_bits",
    "add_pruning_metadata",
    "add_natural_sparsity",
    "hook_inspection",
    "runtime_analysis",
]

TRANSFORM_PASSES = [
    "quantize",
    "summarize_quantization",
    "prune",
    "prune_detach_hook" "conv_bn_fusion",
    "logicnets_fusion",
    "tensorrt_fake_quantize",
    "tensorrt_calibrate",
    "tensorrt_fine_tune",
]

INTERFACE_PASSES = [
    "load_mase_graph",
    "load_node_meta_param",
    "save_mase_graph",
    "save_node_meta_param",
    "tensorrt",
    "onnxruntime",
]

PASSES = {
    # analysis
    "init_metadata": init_metadata_analysis_pass,
    "add_common_metadata": add_common_metadata_analysis_pass,
    "verify_common_metadata": verify_common_metadata_analysis_pass,
    "add_hardware_metadata": add_hardware_metadata_analysis_pass,
    "add_software_metadata": add_software_metadata_analysis_pass,
    "profile_statistics": profile_statistics_analysis_pass,
    "report_graph": report_graph_analysis_pass,
    "report_node_hardware_type": report_node_hardware_type_analysis_pass,
    "report_node_meta_param": report_node_meta_param_analysis_pass,
    "report_node_shape": report_node_shape_analysis_pass,
    "report_node_type": report_node_type_analysis_pass,
    "run_cosim_analysis_pass": run_cosim_analysis_pass,
    "get_synthesis_results": get_synthesis_results,
    "calculate_avg_bits": calculate_avg_bits_mg_analysis_pass,
    "add_pruning_metadata": add_pruning_metadata_analysis_pass,
    "add_natural_sparsity": add_natural_sparsity_metadata_analysis_pass,
    "hook_inspection": hook_inspection_analysis_pass,
    "runtime_analysis": runtime_analysis_pass,
    # interface
    "load_mase_graph": load_mase_graph_interface_pass,
    "load_node_meta_param": load_node_meta_param_interface_pass,
    "save_mase_graph": save_mase_graph_interface_pass,
    "save_node_meta_param": save_node_meta_param_interface_pass,
    "tensorrt": tensorrt_engine_interface_pass,
    "onnxruntime": onnx_runtime_interface_pass,
    # transform
    "quantize": quantize_transform_pass,
    "tensorrt_calibrate": tensorrt_calibrate_transform_pass,
    "tensorrt_fake_quantize": tensorrt_fake_quantize_transform_pass,
    "tensorrt_fine_tune": tensorrt_fine_tune_transform_pass,
    "summarize_quantization": summarize_quantization_analysis_pass,
    "prune": prune_transform_pass,
    "prune_detach_hook": prune_detach_hook_transform_pass,
    # "remove_prune_wrappers": prune_unwrap_transform_pass,
    "conv_bn_fusion": conv_bn_fusion_transform_pass,
    "logicnets_fusion": logicnets_fusion_transform_pass,
    "onnx_annotate": onnx_annotate_transform_pass,
}
