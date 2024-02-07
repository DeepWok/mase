from .base import HWRunnerBase


class ResourceUsageRunner(HWRunnerBase):
    available_metrics = (
        "lut",
        "bram",
        "dsp",
        "ff",
    )

    def _post_init_setup(self) -> None:
        pass

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        metrics = {}
        return metrics
