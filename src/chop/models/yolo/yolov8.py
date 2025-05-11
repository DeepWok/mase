from chop.models.utils import register_mase_model, ModelSource, ModelTaskType
from ultralytics.nn.tasks import DetectionModel, SegmentationModel
from ultralytics import YOLO
import ultralytics.nn.modules as unnmod
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.tal import make_anchors, dist2bbox
import torch
import types


@register_mase_model(
    name="yolov8-detection",
    checkpoints=[
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
    ],
    model_source=ModelSource.VISION_OTHERS,
    task_type=ModelTaskType.VISION,
    image_detection=True,
    is_fx_traceable=True,
)
class MaseYoloDetectionModel(DetectionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_ops = {
            "modules": {},
        }

    def forward(self, x):
        if self.training:
            return super().forward(x)
        # return the bounding boxes directly
        return non_max_suppression(super().forward(x))


@register_mase_model(
    name="yolov8-segmentation",
    checkpoints=[
        "yolov8n-seg.pt",
        "yolov8s-seg.pt",
        "yolov8m-seg.pt",
        "yolov8l-seg.pt",
        "yolov8x-seg.pt",
    ],
    model_source=ModelSource.VISION_OTHERS,
    task_type=ModelTaskType.VISION,
    image_instance_segmentation=True,
    is_fx_traceable=True,
)
class MaseYoloSegmentationModel(SegmentationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_ops = {
            "modules": {},
        }

    def forward(self, x):
        if self.training:
            return super().forward(x)

        # Make sure to run the model head in training
        # mode such that it gives the correct output format
        self.model[-1].training = True
        return super().forward(x)


def c2f_forward(self, x):
    # patched to avoid chunk
    y = self.cv1(x)
    y1, y2 = y[:, : self.c, :, :], y[:, self.c :, :, :]
    y = [y1, y2]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def patch_yolo(model):
    for name, module in model.model.named_children():
        if isinstance(module, unnmod.C2f):
            module.forward = types.MethodType(c2f_forward, module)
    return model


def get_yolo_detection_model(checkpoint):
    model = MaseYoloDetectionModel(cfg=checkpoint.replace(".pt", ".yaml"))
    model = patch_yolo(model)
    if ".pt" in checkpoint:
        umodel = YOLO(checkpoint)
        model.load_state_dict(umodel.model.state_dict())
    return model


def get_yolo_segmentation_model(checkpoint):
    assert "-seg" in checkpoint
    model = MaseYoloSegmentationModel(cfg=checkpoint.replace(".pt", ".yaml"))
    model = patch_yolo(model)
    if ".pt" in checkpoint:
        umodel = YOLO(checkpoint)
        model.load_state_dict(umodel.model.state_dict())
    return model


def postprocess_outputs_no_nms(x):
    dfl = unnmod.DFL(16)
    shape = x[0].shape
    strides = [8, 16, 32]
    x_cat = torch.cat([xi.view(shape[0], 16 * 4 + 80, -1) for xi in x], 2)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(x, strides, 0.5))
    box = x_cat[:, : 16 * 4]
    cls = x_cat[:, 16 * 4 :]
    dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    bboxes_preds = torch.cat((dbox, cls.sigmoid()), 1)
    return bboxes_preds


def postprocess_detection_outputs(x):
    bboxes_preds = postprocess_outputs_no_nms(x)
    return non_max_suppression(bboxes_preds, multi_label=True)


def postprocess_segmentation_outputs(seg_outputs):
    x, mask_coefs, mask_protos = seg_outputs
    bboxes_preds = postprocess_outputs_no_nms(x)
    catted = torch.cat([bboxes_preds, mask_coefs], 1)
    # COCO has 80 classes
    pruned_preds = non_max_suppression(catted, multi_label=True, nc=80)
    return pruned_preds, mask_protos
