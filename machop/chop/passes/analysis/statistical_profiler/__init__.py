import torch

from .activation_profiler import run_activation_profiler
from .utils import profile_to_int_frac_config
from .weight_profiler import run_weight_profiler


def run_statistical_profiler(
    model_name,
    task,
    model: torch.nn.Module,
    data_module,
    dummy_inputs_for_fx,
    config_path: str,
    save_dir,
):
    act_profile = run_activation_profiler(
        model_name,
        task,
        model,
        data_module,
        dummy_inputs_for_fx,
        config_path,
        save_dir,
    )
    weight_profile = run_weight_profiler(
        model=model,
        dummy_inputs_for_fx=dummy_inputs_for_fx,
        config_path=config_path,
        save_dir=save_dir,
    )
    return act_profile | weight_profile
