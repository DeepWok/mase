from chop.ir import MaseGraph
from chop.tools.logger import get_logger

logger = get_logger(__name__)


class AutoPipeline:
    """This is the base class for the AutoPipeline.

    It takes a list of passes and runs them in order.

    The output of each pass is stored in a dictionary and can be accessed by the next pass.
    """

    def __init__(self, pass_list=[]) -> None:
        """Initializes the AutoPipeline.

        Args:
            pass_list (list, optional): List of passes to run. Defaults to [].
        """
        self.pass_list = pass_list
        self.pass_outputs = {}

    def __call__(self, mg: MaseGraph, pass_args: dict, skip_passes: list = []):
        for pass_fn in self.pass_list:
            if pass_fn in skip_passes:
                logger.debug(f"Skipping pass: {pass_fn.__name__}")
                continue
            logger.debug(f"Running pass: {pass_fn.__name__}")
            args = pass_args.get(pass_fn.__name__, {})

            for k, v in args.items():
                if isinstance(v, str) and v.startswith("self/"):
                    args[k] = self.pass_outputs[v[5:]]

            mg, pass_output = pass_fn(mg, pass_args=args)
            self.pass_outputs[pass_fn.__name__] = pass_output

        return mg, self.pass_outputs
