import torch
from chop.tools.plt_wrapper.vision.ultralytics_detection import UltralyticsDetectionWrapper
from chop.tools.plt_wrapper.vision.ultralytics_segmentation import UltralyticsSegmentationWrapper
from chop.models.yolo.yolov8 import get_yolo_detection_model, get_yolo_segmentation_model

def dummy_forward(x):
    batch_size = x.shape[0]
    feat1 = torch.randn(batch_size, 144, 80, 80)
    feat2 = torch.randn(batch_size, 144, 40, 40)
    feat3 = torch.randn(batch_size, 144, 20, 20)
    return [feat1, feat2, feat3]

def test_detection_training_step_runs():
    model = get_yolo_detection_model("yolov8n.pt")
    model.forward = dummy_forward
    wrapper = UltralyticsDetectionWrapper(model=model, dataset_info={})

    batch = {
        "img": torch.randn(2, 3, 640, 640),
        "bboxes": torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.3, 0.3, 0.1, 0.1]
        ]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "batch_idx": torch.tensor([0, 1]),
        "ori_shape": [(640, 640), (640, 640)],
        "ratio_pad": [(1.0, 1.0), (1.0, 1.0)],
    }

    model.eval()
    with torch.no_grad():
        loss = wrapper.training_step(batch, 0)
    assert isinstance(loss.item(), float)

def test_segmentation_training_step_runs():
    model = get_yolo_segmentation_model("yolov8n-seg.pt")
    wrapper = UltralyticsSegmentationWrapper(model=model, dataset_info={})

    batch = {
        "img": torch.randn(2, 3, 640, 640),
        "bboxes": torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.3, 0.3, 0.1, 0.1]
        ]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "batch_idx": torch.tensor([0, 1]),
        "ori_shape": [(640, 640), (640, 640)],
        "ratio_pad": [(1.0, 1.0), (1.0, 1.0)],
        "masks": torch.rand(2, 160, 160),
    }

    model.eval()
    with torch.no_grad():
        loss = wrapper.training_step(batch, 0)
    assert isinstance(loss.item(), float)
