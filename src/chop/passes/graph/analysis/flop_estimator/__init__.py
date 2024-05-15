import torch

from ..utils import _import_config_from_py_file
from .deepspeed import estimate_sw_deepspeed
from .fine_grained import estimate_sw_fine_grained

estimator_style_map = {
    "deepspeed": estimate_sw_deepspeed,
    # FIXME
    "fine-grained": estimate_sw_fine_grained,
}


def run_flop_estimator(
    model_name: int,
    task: str,
    info: dict,
    model: torch.nn.Module,
    data_module,
    config_path: str = None,
    save_dir: str = None,
):
    config = _import_config_from_py_file(model_name, config_path)

    # set default to deepspeed
    if "style" in config:
        estimator_style = config.pop("style")
    else:
        estimator_style = config.get("style", "deepspeed")

    estimator_style_map[estimator_style](
        model_name=model_name,
        info=info,
        model=model,
        task=task,
        data_module=data_module,
        save_dir=save_dir,
        config=config,
    )
