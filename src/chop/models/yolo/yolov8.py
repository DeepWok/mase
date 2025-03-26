from chop.models.utils import register_mase_model, ModelSource, ModelTaskType
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
import ultralytics.nn.modules as unnmod
from ultralytics.utils.ops import non_max_suppression
import torch
import types


@register_mase_model(
    name="yolov8-detection",
    checkpoints=[
        "yolov8n",
    ],
    model_source=ModelSource.VISION_OTHERS,
    task_type=ModelTaskType.VISION,
    image_detection=True,
    is_fx_traceable=True,
)
class MaseYoloDetectionModel(DetectionModel):
    def forward(self, x):
        if self.training:
            return super().forward(x)
        # return the bounding boxes directly
        return non_max_suppression(super().forward(x))


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
