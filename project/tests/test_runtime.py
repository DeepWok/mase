import torch
from chop.models.yolo.yolov8 import postprocess_detection_outputs

def test_postprocess_detection_outputs_runs():
    # 144 = 16*4 + 80 (expected by the view)
    dummy_preds = [
        torch.randn(1, 144, 80, 80),
        torch.randn(1, 144, 40, 40),
        torch.randn(1, 144, 20, 20)
    ]
    output, nms = postprocess_detection_outputs(dummy_preds)
    assert isinstance(output, torch.Tensor)
    assert isinstance(nms, list)
