import torch
from chop.models.yolo.yolov8 import (
    get_yolo_detection_model,
    get_yolo_segmentation_model,
    patch_yolo,
)

def test_detection_model_forward_pass():
    model = get_yolo_detection_model("yolov8n.pt")
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(dummy_input)
    assert isinstance(output, list), "Output should be a list of detections"

def test_segmentation_model_forward_pass():
    model = get_yolo_segmentation_model("yolov8n-seg.pt")
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(dummy_input)
    assert isinstance(output, (list, tuple)), "Segmentation model output should be list or tuple"

def test_patch_yolo_does_not_crash():
    model = get_yolo_detection_model("yolov8n.pt")
    model = patch_yolo(model)
