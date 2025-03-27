import os
import sys
import torch
import torch.nn as nn
import torch.fx as fx
from ultralytics import YOLO

@fx.wrap
def safe_cat(x, dim):
    return torch.cat(tuple(x), dim=dim)

def safe_settatr(obj, name, value):
    if isinstance(value, (int, list, str, float)):
        setattr(obj, name, value)

class FXSafeConcat(nn.Module):
    def __init__(self, orig_module):
        super().__init__()
        for name, value in vars(orig_module).items():
            safe_settatr(self, name, value)
        self.d = orig_module.d

    def forward(self, x):
        return safe_cat(x, dim=self.d)

FXSafeConcat.__module__ = "__main__"

@fx.wrap
def safe_detect(module, *args):
    return module(*args)

class FXSafeDetect(nn.Module):
    def __init__(self, orig_module):
        super().__init__()
        self.orig_module = orig_module
        for name, value in vars(orig_module).items():
            safe_settatr(self, name, value)
    def forward(self, *args):
        return safe_detect(self.orig_module, *args)

@fx.wrap
def safe_c2f(module, x):
    return module(x)

class FXSafeC2f(nn.Module):
    def __init__(self, orig_module):
        super().__init__()
        for name, value in vars(orig_module).items():
            safe_settatr(self, name, value)
        self.orig_module = orig_module
    def forward(self, x, **kwargs):
        return safe_c2f(self.orig_module, x)

def safe_list_create(config, data_in, y):
    print("safe_list_create used")
    def eval_if_module(x):
        return x.forward() if isinstance(x, nn.Module) else x
    return [eval_if_module(config), eval_if_module(data_in), eval_if_module(y)]

def safe_append(data_in, element):
    print("safe_append used")
    if data_in is None:
        data_in = []
    elif not isinstance(data_in, list):
        data_in = list(data_in)
    data_in.append(element)
    return data_in

def safe_unbind(data_in, dim):
    print("safe_unbind used")
    return torch.unbind(data_in, dim=dim)

# Inject safe_* functions into ultralytics.nn.tasks
import ultralytics.nn.tasks as tasks

tasks.safe_list_create = safe_list_create
tasks.safe_append = safe_append
tasks.safe_unbind = safe_unbind

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("yolov8n.pt", task="detect")
quantized_model = torch.load("/home/infres/casta-22/ADLS/mase_calibrated_qat.pt", map_location=device)
model.model = quantized_model

# Patch a dummy 'fuse' method on the GraphModule
if not hasattr(model.model, "fuse"):
    model.model.fuse = lambda verbose=False: model.model

# Patch dummy stride attribute if missing 
if not hasattr(model.model, "stride"):
    model.stride = torch.tensor([8, 16, 32])

# Run prediction on an image
results = model("https://ultralytics.com/images/bus.jpg")
print(results)
