from .base import HWRunnerBase


class LatencyRunner(HWRunnerBase):
    available_metrics = ("latency",)

    def _post_init_setup(self) -> None:
        pass

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        metrics = {}
        return metrics
