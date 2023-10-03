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
    fpgaconvnet_optimiser_analysis_pass,
)
from .transforms import (
    load_mase_graph_transform_pass,
    load_node_meta_param_transform_pass,
    prune_transform_pass,
    prune_unwrap_transform_pass,
    quantize_transform_pass,
    save_mase_graph_transform_pass,
    save_node_meta_param_transform_pass,
    summarize_quantization_analysis_pass,
    conv_bn_fusion_transform_pass,
    logicnets_fusion_transform_pass,
    onnx_annotate_transform_pass,
)
from .transforms.quantize import quantized_func_map, quantized_module_map
from .transforms.quantize.quant_parsers import parse_node_config

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
    "fpgaconvnet_optimiser",
]
TRANSFORM_PASSES = [
    "load_mase_graph",
    "load_node_meta_param",
    "save_mase_graph",
    "save_node_meta_param",
    "quantize",
    "summarize_quantization",
    "prune",
    "remove_prune_wrappers",
    "conv_bn_fusion",
    "logicnets_fusion",
]

PASSES = {
    # analysis
    "init_metadata": init_metadata_analysis_pass,
    "add_common_metadata": add_common_metadata_analysis_pass,
    "add_hardware_metadata": add_hardware_metadata_analysis_pass,
    "add_software_metadata": add_software_metadata_analysis_pass,
    "profile_statistics": profile_statistics_analysis_pass,
    "report_graph": report_graph_analysis_pass,
    "report_node_hardware_type": report_node_hardware_type_analysis_pass,
    "report_node_meta_param": report_node_meta_param_analysis_pass,
    "report_node_shape": report_node_shape_analysis_pass,
    "report_node_type": report_node_type_analysis_pass,
    "fpgaconvnet_optimiser": fpgaconvnet_optimiser_analysis_pass,
    # transform
    "load_mase_graph": load_mase_graph_transform_pass,
    "load_node_meta_param": load_node_meta_param_transform_pass,
    "save_mase_graph": save_mase_graph_transform_pass,
    "save_node_meta_param": save_node_meta_param_transform_pass,
    "quantize": quantize_transform_pass,
    "summarize_quantization": summarize_quantization_analysis_pass,
    "prune": prune_transform_pass,
    "remove_prune_wrappers": prune_unwrap_transform_pass,
    "conv_bn_fusion": conv_bn_fusion_transform_pass,
    "logicnets_fusion": logicnets_fusion_transform_pass,
    "onnx_annotate": onnx_annotate_transform_pass,
}
