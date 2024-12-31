import chop.passes as passes

from .auto_pipeline import AutoPipeline


class CompressionPipeline(AutoPipeline):
    """This pipeline is used for compressing the weights of a model.

    It runs the following passes:

    - init_metadata_analysis_pass

    - add_common_metadata_analysis_pass

    - quantize_transform_pass

    - prune_transform_pass
    """

    def __init__(self) -> None:
        """Initializes the AutoPipeline."""

        pass_list = [
            passes.init_metadata_analysis_pass,
            passes.add_common_metadata_analysis_pass,
            passes.quantize_transform_pass,
            passes.prune_transform_pass,
        ]

        super().__init__([pass_list])
