import chop.passes as passes

from .auto_pipeline import AutoPipeline


class AutoPipelineForEmitVerilog(AutoPipeline):
    def __init__(self) -> None:

        pass_list = [
            passes.init_metadata_analysis_pass,
            passes.report_graph_analysis_pass,
            passes.add_common_metadata_analysis_pass,
            passes.patch_metadata_transform_pass,
            passes.add_hardware_metadata_analysis_pass,
            passes.report_node_meta_param_analysis_pass,
            passes.emit_verilog_top_transform_pass,
            passes.emit_bram_transform_pass,
            passes.emit_internal_rtl_transform_pass,
            passes.emit_cocotb_transform_pass,
            passes.emit_vivado_project_transform_pass,
        ]

        super().__init__(pass_list)
