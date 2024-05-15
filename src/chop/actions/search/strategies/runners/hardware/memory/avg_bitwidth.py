import torch
from ..base import HWRunnerBase
from .model_profiler import get_model_profiler

from chop.passes.module.analysis.quantization import (
    calculate_avg_bits_module_analysis_pass,
)


class RunnerAvgBitwidthManualModel(HWRunnerBase):
    # this is the avg bitwidth estimator of manually quantized model
    available_metrics = ("average_bitwidth", "memory_density")

    def _post_init_setup(self) -> None:
        self.compare_to = self.config["compare_to"]
        self.profiler_name = self.config.get("profiler", "profiler_pass")

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        metrics = {}
        if not isinstance(model, torch.nn.Module):
            pass_model = model.model
        else:
            pass_model = model

        data_loader = data_module.val_dataloader()
        input_ids = next(iter(data_loader))["input_ids"]
        seq_len = input_ids.shape[1]

        if self.profiler_name == "profiler_pass":
            calculate_avg_bits_module_analysis_pass(pass_model, metrics)
            metrics["memory_density"] = self.compare_to / metrics["average_bitwidth"]
        else:
            p_metrics = get_model_profiler(self.profiler_name)(
                pass_model.config, seq_len=seq_len
            )
            metrics["average_bitwidth"] = (
                p_metrics["param_bits"] + p_metrics["act_bits"]
            ) / (p_metrics["num_params"] + p_metrics["num_acts"])
            metrics["memory_density"] = self.compare_to / metrics["average_bitwidth"]

        return metrics


class RunnerAvgBitwidth(HWRunnerBase):
    available_metrics = ("average_bitwidth",)

    def _post_init_setup(self) -> None:
        assert (
            "compare_to" in self.config
        ), "Must specify `compare_to` (integer bitwidth) in config"

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        metrics = {}
        if not isinstance(model, torch.nn.Module):
            pass_model = model.model
        else:
            pass_model = model
        # compose hardware analysis passes here
        _, info = calculate_avg_bits_module_analysis_pass(pass_model, metrics)
        metrics["average_bitwidth"] = info["average_bitwidth"]
        metrics["memory_density"] = self.config["compare_to"] / info["average_bitwidth"]
        return metrics
