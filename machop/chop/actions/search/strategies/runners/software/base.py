class SWRunnerBase:
    """
    Base class for software runner.

    ---

    Check `machop/chop/actions/search/strategies/runners/software/basic.py` for an example.
    """

    available_metrics: tuple[str] = None  # metric names

    def __init__(
        self, model_info, task: str, dataset_info, accelerator, config: dict = None
    ):
        self.model_info = model_info
        self.task = task
        self.dataset_info = dataset_info
        self.accelerator = accelerator

        self.config = config

        self._post_init_setup()

    def _post_init_setup(self) -> None:
        """
        Setup the runner after __init__.
        """
        raise NotImplementedError()

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        """
        Run the model on the data_loader for num_batches, and return a dict of metrics in the form of dict[str, float]
        """
        raise NotImplementedError()
