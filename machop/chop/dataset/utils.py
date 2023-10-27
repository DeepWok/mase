from dataclasses import dataclass
from enum import Enum


class DatasetSource(Enum):
    """
    The source of the dataset, must be one of the following:
    - MANUAL: manual dataset from MASE
    - HF_DATASETS: dataset from HuggingFace datasets
    - TORCHVISION: dataset from torchvision
    - OTHERS: other datasets
    """

    MANUAL = "manual"
    HF_DATASETS = "hf_datasets"
    TORCHVISION = "torchvision"
    OTHERS = "others"


class DatasetSplit(Enum):
    """
    The split of the dataset, must be one of the following:
    - TRAIN: training split
    - VALIDATION: validation split
    - TEST: test split
    - PRED: prediction split
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PRED = "pred"


@dataclass
class MaseDatasetInfo:
    """
    The dataset info for MASE.
    """

    name: str

    # dataset source
    dataset_source: DatasetSource

    # available splits
    available_splits: tuple[DatasetSplit]

    # requires preprocessing
    requires_preprocessing: bool = False
    preprocess_one_split_for_all: bool = True

    # tasks
    # cls
    image_classification: bool = False

    # cls
    physical_data_point_classification: bool = False

    # cls
    sequence_classification: bool = False
    # lm
    causal_LM: bool = False
    # tran
    seq2seqLM: bool = False

    # classification fields
    num_classes: int = None
    image_size: tuple[int] = None
    num_features: int = None

    def __post_init__(self):
        self.dataset_source = (
            DatasetSource(self.dataset_source)
            if isinstance(self.dataset_source, str)
            else self.dataset_source
        )
        self.available_splits = tuple(
            DatasetSplit(split) if isinstance(split, str) else split
            for split in self.available_splits
        )
        self._entries = {
            "name",
            "dataset_source",
            "available_splits",
            "image_classification",
            "physical_data_point_classification",
            "sequence_classification",
            "causal_LM",
            "seq2seqLM",
            "num_classes",
            "image_size",
            "num_features",
        }

    @property
    def train_split_available(self):
        return DatasetSplit.TRAIN in self.available_splits

    @property
    def validation_split_available(self):
        return DatasetSplit.VALID in self.available_splits

    @property
    def test_split_available(self):
        return DatasetSplit.TEST in self.available_splits

    @property
    def pred_split_available(self):
        return DatasetSplit.PRED in self.available_splits

    def __getitem__(self, key: str):
        if key in self._entries:
            return getattr(self, key)
        else:
            raise KeyError(f"key {key} not found in MaseDatasetInfo")


def add_dataset_info(
    name: str,
    dataset_source: DatasetSource,
    available_splits: tuple[DatasetSplit],
    image_classification: bool = False,
    physical_data_point_classification: bool = False,
    sequence_classification: bool = False,
    causal_LM: bool = False,
    seq2seqLM: bool = False,
    num_classes: int = None,
    image_size: tuple[int] = None,
    num_features: int = None,
):
    """
    a decorator (factory) for adding dataset info to a dataset class

    Args:
        name (str): the name of the dataset
        dataset_source (DatasetSource): the source of the dataset, must be one of "manual", "hf_datasets", "torchvision", "others"
        available_splits (tuple[DatasetSplit]): a tuple of the available splits of the dataset, the split must be one of "train", "valid", "test", "pred"
        image_classification (bool, optional): whether the dataset is for image classification. Defaults to False.
        physical_data_point_classification (bool, optional):  whether the dataset is for physical classification. Defaults to False.
        sequence_classification (bool, optional): whether the dataset is for sequence classification. Defaults to False.
        causal_LM (bool, optional): whether the dataset is for causal language modeling. Defaults to False.
        seq2seqLM (bool, optional): whether the dataset is for sequence-to-sequence language modeling. Defaults to False.
        num_classes (int, optional): the number of classes of the dataset. Defaults to None.
        image_size (tuple[int], optional): the image size of the dataset. Defaults to None.
        num_features (int, optional): Specifies the number of features in the dataset. This is particularly relevant for physical classification tasks that involve input feature vectors. Defaults to None.

    Returns:
        type: the dataset class with dataset info
    """

    def _add_dataset_info_to_cls(cls: type):
        cls.info = MaseDatasetInfo(
            name=name,
            dataset_source=dataset_source,
            available_splits=available_splits,
            image_classification=image_classification,
            physical_data_point_classification=physical_data_point_classification,
            sequence_classification=sequence_classification,
            causal_LM=causal_LM,
            seq2seqLM=seq2seqLM,
            num_classes=num_classes,
            image_size=image_size,
            num_features=num_features,
        )

        return cls

    return _add_dataset_info_to_cls
