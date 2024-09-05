import torch.distributed as dist

import chop.passes as passes
from chop.tools import get_logger

from .auto_pipeline import AutoPipeline

logger = get_logger(__name__)
logger.setLevel("INFO")


class AutoPipelineForDistributedInference(AutoPipeline):
    """This pipeline is used for distributed inference.

    It runs the following pre-processing passes:

    - replace_method_with_function

    Then, it raises the graph to Mase IR:

    - init_metadata_analysis_pass

    - report_graph_analysis_pass

    - add_common_metadata_analysis_pass

    Then, it runs the following passes:

    - autosharding_analysis_pass

    If the distributed setup is initialized, it runs the following passes:

    - insert_dtensor_wrapper_transform_pass

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
            passes.report_graph_analysis_pass,
            passes.add_common_metadata_analysis_pass,
        ]

        # Autosharding
        pass_list += [
            passes.autosharding_analysis_pass,
        ]

        # Only run the following in distributed setup
        if dist.is_initialized():
            pass_list += [
                passes.insert_dtensor_wrapper_transform_pass,
                passes.resharding_transform_pass,
            ]
        else:
            logger.info(
                "Torch distributed is not initialized, so will skip the following passes: insert_dtensor_wrapper_transform_pass, resharding_transform_pass"
            )

        super().__init__(pass_list)
