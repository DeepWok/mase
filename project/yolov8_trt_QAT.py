# %%

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
from chop import MaseGraph
from chop.models.yolo.yolov8 import get_yolo_detection_model


# Figure out the correct path
machop_path = Path(".").resolve().parent.parent.parent
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

# Add directory to the PATH so that chop can be called
new_path = "../../.."
full_path = os.path.abspath(new_path)
os.environ["PATH"] += os.pathsep + full_path

import chop.passes as passes
from chop.passes.graph.analysis.add_metadata.common_metadata_layers import func_data
from chop.tools.logger import set_logging_verbosity
from chop.tools.get_input import InputGenerator
from chop.passes.graph import (
    summarize_quantization_analysis_pass,
    tensorrt_calibrate_transform_pass,
    tensorrt_fake_quantize_transform_pass,
    tensorrt_engine_interface_pass,
)

set_logging_verbosity("info")

from chop.passes.graph.transforms.tensorrt.quantize.fine_tune import (
    tensorrt_fine_tune_transform_pass,
)
from chop.dataset import MaseDataModule


# %%
# Load a pretrained YOLO model
model_name = "yolov8n.pt"
model = get_yolo_detection_model(model_name)
# model = get_yolo_segmentation_model("yolov8m-seg.pt")


# Define a safe wrapper for torch.cat to avoid tracing its internals
@fx.wrap
def safe_cat(x, dim):
    return torch.cat(tuple(x), dim=dim)


def safe_settatr(obj, name, value):
    if isinstance(value, int):
        setattr(obj, name, value)
    elif isinstance(value, list):
        setattr(obj, name, value)
    elif isinstance(value, str):
        setattr(obj, name, value)
    elif isinstance(value, float):
        setattr(obj, name, value)


# FX-safe wrapper for Concat
class FXSafeConcat(nn.Module):
    def __init__(self, orig_module):
        super().__init__()
        attrs = vars(orig_module)
        for name, value in attrs.items():
            safe_settatr(self, name, value)
        self.d = orig_module.d

    def forward(self, x):
        return safe_cat(x, dim=self.d)


# Define a safe wrapper for Detect module calls
@fx.wrap
def safe_detect(
    module,
    *args,
):
    return module(
        *args,
    )


# Replace problematic modules in the model with FX-safe versions
for name, module in model.model.named_children():
    if isinstance(module, ultralytics.nn.modules.conv.Concat):
        print(f"Replacing module {name} with FXSafeConcat")
        setattr(model.model, name, FXSafeConcat(module))


cf_args = {
    # "x": torch.randn(1, 3, 640, 640),
    # "profile": False,
    # "visualize": False,
    # "augment": False,
    # "embed": None,
}

mg = MaseGraph(model, cf_args=cf_args)

# Set custom_ops
CUSTOM_OPS = {
    "modules": {
        FXSafeConcat: "",
    },
    # "functions": {safe_cat: "", safe_detect: "", },
}
setattr(mg.model, "custom_ops", CUSTOM_OPS)


mg.model.patched_op_names = [
    "safe_cat",
    "safe_detect",
    "safe_settatr",
    "safe_list_create",
    "safe_append",
    "safe_unbind",
]

# %%
func_data["safe_detect"] = {"module": "detect", "input": "data_in"}
func_data["safe_cat"] = {"module": "concat", "input": "data_in", "dim": "config"}
func_data["safe_settatr"] = {
    "module": "settatr",
    "input": "data_in",
    "name": "config",
    "value": "config",
}
func_data["safe_list_create"] = {
    "module": "list_create",
    "m": "config",
    "x": "data_in",
    "y": "config",
}
func_data["safe_append"] = {
    "module": "append",
    "x": "data_in",
}
func_data["safe_unbind"] = {
    "module": "unbind",
    "x": "data_in",
    "dim": "config",
}

param = next(mg.model.model.parameters())[1]

dummy_input = torch.rand(1, 3, 640, 640, dtype=param.dtype).to(param.device)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    {
        "dummy_in": {
            "x": dummy_input,
            "profile_1": False,
            "visualize_1": False,
            "augment_1": False,
            "embed_1": None,
        },
        "add_value": True,
    },
)
mg, _ = passes.add_software_metadata_analysis_pass(mg, None)

# %%


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

# %%
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
mg, _ = tensorrt_calibrate_transform_pass(new_mg, pass_args=tensorrt_config)
mg.export("mase_calibrated")

# %%
mg, _ = tensorrt_fine_tune_transform_pass(mg, pass_args=tensorrt_config)
mg.export("mase_calibrated_qat")

# %%
# mg, meta = tensorrt_engine_interface_pass(mg, pass_args=tensorrt_config)

# %%
