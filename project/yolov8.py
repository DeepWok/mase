# %%
import logging
from ultralytics import YOLO
from chop import MaseGraph
import torch.fx as fx
import torch.nn as nn
import torch
import ultralytics

# Load a pretrained YOLO model
model = YOLO("yolov8n.yaml")  # Choose the appropriate model


# Define a safe wrapper for torch.cat to avoid tracing its internals
@fx.wrap
def safe_cat(x, dim):
    print("Number of vec ", len(x), "dim", dim)
    for v in x:
        print("Shape ", v.shape)
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


# FX-safe wrapper for Detect
class FXSafeDetect(nn.Module):
    def __init__(self, orig_module):
        super().__init__()
        self.orig_module = orig_module
        # Instantiate all the original module's parameters
        attrs = vars(orig_module)
        for name, value in attrs.items():
            safe_settatr(self, name, value)

    def forward(self, *args):
        return (safe_detect(self.orig_module, *args),)


# Define a safe wrapper for C2f module calls
@fx.wrap
def safe_c2f(
    module,
    x,
):
    return module(x)


# FX-safe wrapper for C2f
class FXSafeC2f(nn.Module):
    def __init__(self, orig_module):
        super().__init__()

        # Instantiate all the original module's parameters
        attrs = vars(orig_module)
        for name, value in attrs.items():
            safe_settatr(self, name, value)

        self.orig_module = orig_module

    def forward(self, x, **kwargs):
        return safe_c2f(self.orig_module, x)


# Replace problematic modules in the model with FX-safe versions
for name, module in model.model.model.named_children():
    if isinstance(module, ultralytics.nn.modules.conv.Concat):
        print(f"Replacing module {name} with FXSafeConcat")
        setattr(model.model.model, name, FXSafeConcat(module))
    elif isinstance(module, ultralytics.nn.modules.head.Detect):
        print(f"Replacing module {name} with FXSafeDetect")
        setattr(model.model.model, name, FXSafeDetect(module))
    elif isinstance(module, ultralytics.nn.modules.block.C2f):
        print(f"Replacing module {name} with FXSafeC2f")
        setattr(model.model.model, name, FXSafeC2f(module))


cf_args = {
    "x": torch.randn(1, 3, 640, 640),
    # "profile": False,
    # "visualize": False,
    # "augment": False,
    # "embed": None,
}

# %%

mg = MaseGraph(model.model, cf_args=cf_args)
# Set custom_ops
CUSTOM_OPS = {
    "modules": {FXSafeC2f: "", FXSafeConcat: "", FXSafeDetect: ""},
    # "functions": {safe_cat: "", safe_detect: "", safe_c2f: ""},
}
setattr(mg.model, "custom_ops", CUSTOM_OPS)

print("MaseGraph successfully created:")
print(mg)

mg.model.patched_op_names = [
    "safe_cat",
    "safe_detect",
    "safe_c2f",
    "safe_settatr",
    "safe_list_create",
    "safe_append",
    "safe_unbind",
]

# %%

import chop.passes as passes
from chop.passes.graph.analysis.add_metadata.common_metadata_layers import func_data

# Add metadata for the custom functions
func_data["safe_c2f"] = {
    "module": "c2f",
    "input": "data_in",
    "c1": "config",
    "c2": "config",
    "shortcut": False,
    "g": "config",
    "e": "config",
}
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

print(mg.fx_graph)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    {
        "dummy_in": {
            "x_1": dummy_input,
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
