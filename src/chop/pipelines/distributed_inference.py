import chop.passes as passes

from .auto_pipeline import AutoPipeline


class AutoPipelineForDistributedInference(AutoPipeline):
    def __init__(self) -> None:

        pass_list = [
            passes.init_metadata_analysis_pass,
            passes.report_graph_analysis_pass,
            passes.add_common_metadata_analysis_pass,
            passes.report_node_meta_param_analysis_pass,
            passes.autosharding_analysis_pass,
            passes.resharding_transform_pass,
            passes.graph.analysis.report.report_parallelization_analysis_pass,
        ]

        super().__init__(pass_list)