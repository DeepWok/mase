import torch

# from .flop_estimator import run_flop_estimator
# from .statistical_profiler import run_statistical_profiler
from .add_metadata import add_common_metadata_analysis_pass, add_mase_ops_analysis_pass
from .init_metadata import init_metadata_analysis_pass
from .report import report_node_type_analysis_pass, report_node_shape_analysis_pass
from .verify import (
    verify_common_metadata_analysis_pass,
    verify_hardware_metadata_analysis_pass,
    verify_metadata_analysis_pass,
    verify_software_metadata_analysis_pass,
)

analysis_passes = {
    "init_metadata": init_metadata_analysis_pass,
    "add_common_metadata": add_common_metadata_analysis_pass,
    "add_mase_ops": add_mase_ops_analysis_pass,
}

# def run_sw_estimator(
#     estimate_sw: str,
#     model_name: int,
#     task: str,
#     info: dict,
#     model: torch.nn.Module,
#     data_module,
#     config_path: str,
#     dummy_inputs_for_fx: dict,
#     save_dir: str,
# ):
#     if estimate_sw in ["stat", "statistical"]:
#         run_statistical_profiler(
#             model_name=model_name,
#             task=task,
#             model=model,
#             data_module=data_module,
#             dummy_inputs_for_fx=dummy_inputs_for_fx,
#             config_path=config_path,
#             save_dir=save_dir,
#         )
#     elif estimate_sw in ["flop"]:
#         run_flop_estimator(
#             model_name=model_name,
#             task=task,
#             info=info,
#             model=model,
#             data_module=data_module,
#             config_path=config_path,
#             save_dir=save_dir,
#         )
#     else:
#         raise RuntimeError(f"Unsupported `--estimate-sw` ({estimate_sw})")
