import torch

from .activation_profiler import run_activation_profiler


def run_statistical_profiler(
    model_name,
    task,
    model: torch.nn.Module,
    data_module,
    dummy_inputs_for_fx,
    config_path: str,
    save_dir,
):
    run_activation_profiler(
        model_name,
        task,
        model,
        data_module,
        dummy_inputs_for_fx,
        config_path,
        save_dir,
    )
