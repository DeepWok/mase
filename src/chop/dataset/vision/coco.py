import os
from ..utils import add_dataset_info
from ultralytics.data.dataset import YOLODataset
import ultralytics
import time
from pathlib import Path

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    ROOT,
    SETTINGS_FILE,
    TQDM,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
    yaml_load,
    yaml_save,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes


def check_det_dataset(yaml_path, dataset_path, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        path : Path to the dataset. (Overrides the path in the dataset descriptor yaml file)
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """
    file = check_file(yaml_path)
    # Read YAML
    data = yaml_load(file, append_filename=True)  # dictionary

    # Checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(
                        f"{yaml_path} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."
                    )
                )
            LOGGER.info(
                "WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format."
            )
            data["val"] = data.pop(
                "validation"
            )  # replace 'validation' key with 'val' key
    if "names" not in data and "nc" not in data:
        raise SyntaxError(
            emojis(
                f"{yaml_path} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."
            )
        )
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(
            emojis(
                f"{yaml_path} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."
            )
        )
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])

    # Set paths
    for k in "train", "val", "test", "minival":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (dataset_path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (dataset_path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((dataset_path / x).resolve()) for x in data[k]]

    # Parse YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [
            Path(x).resolve() for x in (val if isinstance(val, list) else [val])
        ]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(yaml_path)  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_FILE}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # success
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = os.system(s)
            else:  # python script
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = (
                f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}"
                if r in {0, None}
                else f"failure {dt} ❌"
            )
            LOGGER.info(f"Dataset download {s}\n")
    check_font(
        "Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf"
    )  # download fonts

    return data  # dictionary


@add_dataset_info(
    name="coco",
    dataset_source="others",  # "ultralytics",
    available_splits=("train", "validation"),
    image_classification=False,
    num_classes=80,
    image_size=(3, 640, 640),
)
class CocoMase(YOLODataset):
    def __init__(
        self,
        root: os.PathLike,
        img_path: os.PathLike,
        task_name: str,
        download: bool = True,
    ) -> None:
        # Find COCO yaml file in ultralytics
        ULTRALYTICS_PATH = ultralytics.__path__[0]
        coco_yaml_path = f"{ULTRALYTICS_PATH}/cfg/datasets/coco.yaml"
        coco = check_det_dataset(
            yaml_path=coco_yaml_path, dataset_path=root, autodownload=download
        )
        super().__init__(data=coco, img_path=img_path, imgsz=640, task=task_name)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


def get_coco_detection_dataset(
    task_name: str,
    path: os.PathLike,
    train: bool,
):
    if train:
        img_path = os.path.join(path, "images", "train2017")
    else:
        img_path = os.path.join(path, "images", "val2017")
    return CocoMase(
        root=path,
        img_path=img_path,
        task_name=task_name,
        download=True,
    )
