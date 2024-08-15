import chop.passes as passes

from .auto_pipeline import AutoPipeline


class AutoPipelineForDistributedInference(AutoPipeline):
    """This pipeline is used for distributed inference.

    It runs the following passes:

    - init_metadata_analysis_pass

    - report_graph_analysis_pass

    - add_common_metadata_analysis_pass

    - autosharding_analysis_pass

    - resharding_transform_pass
    """

    def __init__(self) -> None:
        """Initializes the AutoPipeline."""

        # Pre-processing
        pass_list = [
            passes.replace_method_with_function,
        ]

        # Raise to Mase IR
        pass_list += [
            passes.init_metadata_analysis_pass,
            passes.add_common_metadata_analysis_pass,
        ]

        # Autosharding
        pass_list += [
            passes.autosharding_analysis_pass,
        ]

        # Only run the following in distributed setup
        pass_list += [
            passes.insert_dtensor_wrapper_transform_pass,
            passes.resharding_transform_pass,
        ]

        super().__init__(pass_list)
