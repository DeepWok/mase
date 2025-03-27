import torch
from chop.models.yolo.yolov8 import postprocess_detection_outputs, postprocess_segmentation_outputs

def test_postprocess_detection_outputs_runs():
    # 144 = 16*4 + 80 (expected by the view)
    dummy_preds = [
        torch.randn(1, 144, 80, 80),
        torch.randn(1, 144, 40, 40),
        torch.randn(1, 144, 20, 20)
    ]
    nms = postprocess_detection_outputs(dummy_preds)
    assert isinstance(nms, list)
    assert isinstance(nms[0], torch.Tensor)
    assert nms[0].shape[1] == 6

def test_postprocess_segmentation_outputs_runs():
    # 144 = 16*4 + 80 (expected by the view)
    dummy_preds = [
        torch.randn(1, 144, 80, 80),
        torch.randn(1, 144, 40, 40),
        torch.randn(1, 144, 20, 20)
    ]
    dummy_mask_coefs = torch.randn(1, 32, 8400)
    dummy_mask_proto = torch.randn(1, 32, 160, 160)
    preds, proto = postprocess_segmentation_outputs((dummy_preds, dummy_mask_coefs, dummy_mask_proto))
    assert isinstance(preds, list)
    assert isinstance(preds[0], torch.Tensor)
    assert preds[0].shape[1] == 38
    assert (proto == dummy_mask_proto).sum().all()
