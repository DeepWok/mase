class HWRunnerBase:
    available_metrics: tuple[str] = None  # metric names

    def __init__(self, model_info, task: str, dataset_info, accelerator):
        self.model_info = model_info
        self.task = task
        self.dataset_info = dataset_info
        self.accelerator = accelerator

    def __call__(
        self, data_loader, model, sampled_config, num_batches: int
    ) -> dict[str, float]:
        """
        Run the model on the data_loader for num_batches, and return a dict of metrics in the form of dict[str, float]
        """
        raise NotImplementedError()
