from chop.ir import MaseGraph
from chop.tools.logger import get_logger

logger = get_logger(__name__)


class AutoPipeline:
    """This is the base class for the AutoPipeline.

    It takes a list of passes and runs them in order.

    The output of each pass is stored in a dictionary and can be accessed by the next pass.
    """

    def __init__(
        self,
        pass_groups=None,
        run_training: bool = False,
    ) -> None:
        """Initializes the AutoPipeline.

        Args:
            pass_list (list, optional): List of passes to run. Defaults to [].
        """
        self.pass_groups = pass_groups if pass_groups is not None else []
        self.pass_outputs = [{}] * len(pass_groups)

    def _run_pass_group(
        self,
        mg: MaseGraph,
        pass_group: list,
        pass_args: dict,
        skip_passes: list = [],
    ):
        pass_outputs = {}

        for pass_fn in pass_group:

            # Check if need to skip this pass
            if pass_fn in skip_passes:
                logger.debug(f"Skipping pass: {pass_fn.__name__}")
                continue

            # Extract pass arguments
            logger.debug(f"Running pass: {pass_fn.__name__}")
            args = pass_args.get(pass_fn.__name__, {})

            # Replace self/ references with values from previous passes
            for k, v in args.items():
                if isinstance(v, str) and v.startswith("self/"):
                    args[k] = pass_outputs[v[5:]]

            mg, pass_output = pass_fn(mg, pass_args=args)
            pass_outputs[pass_fn.__name__] = pass_output

        return mg, pass_outputs

    def __call__(
        self,
        mg: MaseGraph,
        pass_args: dict,
        skip_passes: list = [],
    ):

        for idx, pass_group in enumerate(self.pass_groups):

            logger.debug(f"Running pass group {idx}/{len(self.pass_groups)}.")
            logger.debug(
                f"The following passes will be executed: {[pass_fn.__name__ for pass_fn in pass_group]}"
            )

            mg, pass_outputs = self._run_pass_group(
                mg,
                pass_group,
                pass_args,
                skip_passes,
            )

            self.pass_outputs[idx] = pass_outputs

        return mg, self.pass_outputs
