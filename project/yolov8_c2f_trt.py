# %%
import logging
from ultralytics import YOLO
from chop import MaseGraph
import torch.fx as fx
import torch.nn as nn
import torch
import ultralytics
from chop import passes
import sys
import os
from pathlib import Path
import toml

# Figure out the correct path
machop_path = Path(".").resolve().parent.parent.parent
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

# Add directory to the PATH so that chop can be called
new_path = "../../.."
full_path = os.path.abspath(new_path)
os.environ["PATH"] += os.pathsep + full_path

from chop.tools.utils import to_numpy_if_tensor
from chop.tools.logger import set_logging_verbosity
from chop.tools import get_cf_args, get_dummy_input
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph
from chop.models import get_model_info, get_model, get_tokenizer
from chop.dataset import MaseDataModule, get_dataset_info
from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    tensorrt_calibrate_transform_pass,
    tensorrt_fake_quantize_transform_pass,
    tensorrt_engine_interface_pass,
    runtime_analysis_pass,
)

set_logging_verbosity("info")

from chop.passes.graph.transforms.tensorrt.quantize.fine_tune import (
    tensorrt_fine_tune_transform_pass,
)

set_logging_verbosity("info")

from yolov8_c2f_tracing import mg

HERE = os.path.dirname(os.path.abspath(__file__))
TOML_PATH = f"{HERE}/yolov8_INT8_quantization_by_type.toml"
# Reading TOML file and converting it into a Python dictionary
with open(TOML_PATH, "r") as toml_file:
    pass_args = toml.load(toml_file)

# Extract the 'passes.tensorrt' section and its children
tensorrt_config = pass_args.get("passes", {}).get("tensorrt", {})
# Extract the 'passes.runtime_analysis' section and its children
runtime_analysis_config = pass_args.get("passes", {}).get("runtime_analysis", {})

model_name = pass_args["model"]
dataset_name = pass_args["dataset"]
max_epochs = pass_args["max_epochs"]
batch_size = pass_args["batch_size"]
learning_rate = pass_args["learning_rate"]
accelerator = pass_args["accelerator"]

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

# Add the data_module and other necessary information to the configs
configs = [tensorrt_config, runtime_analysis_config]
for config in configs:
    config["task"] = pass_args["task"]
    config["dataset"] = pass_args["dataset"]
    config["batch_size"] = pass_args["batch_size"]
    config["model"] = pass_args["model"]
    config["data_module"] = data_module
    config["accelerator"] = (
        "cuda" if pass_args["accelerator"] == "gpu" else pass_args["accelerator"]
    )
    if config["accelerator"] == "gpu":
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"


from chop.models.utils import MaseModelInfo, ModelSource, ModelTaskType

model_info = MaseModelInfo(
    name=pass_args["model"],
    model_source=ModelSource.MANUAL,
    task_type=ModelTaskType.VISION,
    image_classification=True,
)
input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="detection",
    which_dataloader="train",
)

new_mg, _ = tensorrt_fake_quantize_transform_pass(graph=mg, pass_args=tensorrt_config)
summarize_quantization_analysis_pass(
    new_mg, {"save_dir": "trt_fake_quantize_summary", "original_graph": mg}
)
# %%
tensorrt_config["dataset_input_field"] = "img"
mg, _ = tensorrt_calibrate_transform_pass(mg, pass_args=tensorrt_config)

# %%
mg, _ = tensorrt_fine_tune_transform_pass(mg, pass_args=tensorrt_config)

# %%
mg, meta = tensorrt_engine_interface_pass(mg, pass_args=tensorrt_config)
