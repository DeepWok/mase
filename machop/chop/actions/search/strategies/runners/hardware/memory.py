import torch
from .base import HWRunnerBase
from ......passes.analysis.total_bits_estimator import total_bits_module_analysis_pass


class RunnerAvgBitwidth(HWRunnerBase):
    available_metrics = ("average_bitwidth",)

    def __call__(
        self, data_loader, model, sampled_config, num_batches: int
    ) -> dict[str, float]:
        metrics = {}
        if not isinstance(model, torch.nn.Module):
            pass_model = model.model
        else:
            pass_model = model
        # compose hardware analysis passes here
        total_bits_module_analysis_pass(pass_model, metrics)
        return metrics
