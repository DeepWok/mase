import torch

from .deepspeed import estimate_sw_deepspeed
from .fine_grained import estimate_sw_fine_grained
from .utils import _import_config_from_py_file

estimator_style_map = {
    "deepspeed": estimate_sw_deepspeed,
    "fine-grained": estimate_sw_fine_grained,
}


def run_estimator(
    model_name: int,
    info: dict,
    model: torch.nn.Module,
    task: str,
    data_loader,
    save_path: str = None,
    config_path: str = None,
):
    config = _import_config_from_py_file(model_name, config_path)

    # set default to deepspeed
    if "style" in config:
        estimator_style = config.pop("style")
    else:
        estimator_style = config.get("style", "deepspeed")

    estimator_style_map[estimator_style](
        model_name, info, model, task, data_loader, save_path, config
    )
