import torch
import torch.nn as nn
import torch.fx as fx
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.conv as conv

# --- Wrappers FX compatibles correctement définis ---

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

def safe_list_create(config, data_in, y):
    return [config, data_in, y]

def safe_append(data_in, element):
    if data_in is None:
        data_in = []
    elif not isinstance(data_in, list):
        data_in = list(data_in)
    data_in.append(element)
    return data_in

def safe_unbind(data_in, dim):
    return torch.unbind(data_in, dim=dim)

# Injection 
tasks.safe_list_create = safe_list_create
tasks.safe_append = safe_append
tasks.safe_unbind = safe_unbind

original_model = YOLO("yolov8n.pt", task="detect")

quantized_model = torch.load("mase_calibrated_qat.pt")

class WrappedQuantizedYOLO(nn.Module):
    def __init__(self, quantized_model, original_wrapper):
        super().__init__()
        self.model = quantized_model
        self.names = original_wrapper.names
        self.stride = original_wrapper.stride
        self.task = original_wrapper.task

    def forward(self, x, augment=False, visualize=False, embed=None):
        return self.model(x)

    def fuse(self, verbose=False):
        return self

original_model.model = WrappedQuantizedYOLO(quantized_model, original_model.model)

results = original_model("https://ultralytics.com/images/bus.jpg")

results[0].show()
print(results[0].boxes)