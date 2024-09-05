from .pruning import prune_transform_pass, prune_detach_hook_transform_pass
from .quantize import quantize_transform_pass, summarize_quantization_analysis_pass
from .verilog import (
    emit_bram_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_logicnets_transform_pass,
    emit_mlir_hls_transform_pass,
    emit_cocotb_transform_pass,
    emit_verilog_top_transform_pass,
    emit_vivado_project_transform_pass,
)
from .utils import (
    conv_bn_fusion_transform_pass,
    logicnets_fusion_transform_pass,
    onnx_annotate_transform_pass,
    metadata_value_type_cast_transform_pass,
)

from .dse import partition_to_multi_device_transform_pass

from .granularity import raise_granularity_transform_pass
from .patching import patch_metadata_transform_pass

from .resharding import resharding_transform_pass
from .insert_dtensor_wrapper import insert_dtensor_wrapper_transform_pass

from .find_replace.method_to_function import replace_method_with_function
