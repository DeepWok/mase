from ultralytics import YOLO
from chop import MaseGraph
import torch.fx as fx
import torch.nn as nn
import torch
import ultralytics
import os

HERE = os.path.dirname(os.path.abspath(__file__))

models_list = open(f"{HERE}/models", "r").readlines()
models_list = [model.strip() for model in models_list]

tracable_models = []
untracable_models = []

for model_name in models_list:
    # Load a pretrained YOLOv5 model
    try:
        model = YOLO(
            model_name
        )  # Choose yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt
    except Exception as e:
        continue
    is_tracable = True
    if False:

        class FXSafeConcat(nn.Module):
            def forward(self, *inputs):
                return torch.cat(inputs, dim=1)  # Avoids dynamic list iteration

        # Replace Concat in the model
        for name, module in model.model.model.named_children():
            if isinstance(module, ultralytics.nn.modules.conv.Concat):
                print(f"Replacing {name} with FXSafeConcat")
                setattr(model.model.model, name, FXSafeConcat())
    # Print the model architecture
    untraceable_modules = []
    traced_modules = []

    for name, module in model.model.model.named_children():
        try:
            traced_submodule = fx.symbolic_trace(module)
            # print(f"✅ {type(module)} is traceable")
            traced_modules.append(str(type(module)))
        except Exception as e:
            # print(f"❌ {type(module)} ({name}) is NOT traceable: {e}")
            untraceable_modules.append(str(type(module)))
            is_tracable = False

    if is_tracable:
        tracable_models.append(model_name)
    else:
        untracable_models.append(model_name)


# Remove duplicate modules
traced_modules = list(set(traced_modules))
untraceable_modules = list(set(untraceable_modules))

if False:

    print("Traced modules : ")
    for module in traced_modules:
        print(module)

    print("Untraceable modules : ")
    for module in untraceable_modules:
        print(module)

    print("Tracable models : ")
    for model in tracable_models:
        print(model)

    print("Untracable models : ")
    for model in untracable_models:
        print(model)


for model_name in tracable_models:
    model = YOLO(model_name)
    try:
        MaseGraph(model.model, cf_args={"dynamic": True})
        print(f"✅ {model_name} is MaseGraph compatible")
    except Exception as e:
        print(f"❌ {model_name} is NOT MaseGraph compatible: {e}")
