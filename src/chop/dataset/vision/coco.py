import os
from ..utils import add_dataset_info
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset
import ultralytics


@add_dataset_info(
    name="coco-detection",
    dataset_source="others",  # "ultralytics",
    available_splits=("train", "validation"),
    image_classification=False,
    num_classes=80,
    image_size=(3, 640, 640),
)
class CocoDetectionMase(YOLODataset):
    def __init__(
        self,
        root: os.PathLike,
        train: bool = True,
        download: bool = True,
    ) -> None:
        # Find COCO yaml file in ultralytics
        ULTRALYTICS_PATH = ultralytics.__path__[0]
        coco_yaml_path = f"{ULTRALYTICS_PATH}/cfg/datasets/coco.yaml"
        coco = check_det_dataset(coco_yaml_path, autodownload=download)
        super().__init__(data=coco, img_path=root, imgsz=640)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


def get_coco_detection_dataset(
    name: str,
    path: os.PathLike,
    train: bool,
):
    return CocoDetectionMase(
        root=path,
        train=train,
        download=True,
    )
