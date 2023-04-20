import torch

from .flop_estimator import run_flop_estimator
from .statistical_profiler import run_statistical_profiler


def run_sw_estimator(
    estimate_sw: str,
    model_name: int,
    task: str,
    info: dict,
    model: torch.nn.Module,
    data_module,
    config_path: str,
    dummy_inputs_for_fx: dict,
    save_dir: str,
):
    if estimate_sw in ["stat", "statistical"]:
        run_statistical_profiler(
            model_name=model_name,
            task=task,
            model=model,
            data_module=data_module,
            dummy_inputs_for_fx=dummy_inputs_for_fx,
            config_path=config_path,
            save_dir=save_dir,
        )
    elif estimate_sw in ["flop"]:
        run_flop_estimator(
            model_name=model_name,
            task=task,
            info=info,
            model=model,
            data_module=data_module,
            config_path=config_path,
            save_dir=save_dir,
        )
    else:
        raise RuntimeError(f"Unsupported `--estimate-sw` ({estimate_sw})")
