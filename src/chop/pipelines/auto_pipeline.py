from chop.ir import MaseGraph
from chop.tools.logger import get_logger

logger = get_logger(__name__)


class AutoPipeline:
    def __init__(self, pass_list=[]) -> None:
        self.pass_list = pass_list

    def __call__(self, mg: MaseGraph, pass_args: dict, skip_passes: list = []):
        for pass_fn in self.pass_list:
            if pass_fn in skip_passes:
                logger.debug(f"Skipping pass: {pass_fn.__name__}")
                continue
            logger.debug(f"Running pass: {pass_fn.__name__}")
            args = pass_args.get(pass_fn.__name__, {})
            mg, _ = pass_fn(mg, pass_args=args)
