import logging
import subprocess
from pathlib import Path

import h5py
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ..utils import add_dataset_info

logger = logging.getLogger(__name__)


JSC_CONFIG = {
    "Inputs": [
        "j_zlogz",
        "j_c1_b0_mmdt",
        "j_c1_b1_mmdt",
        "j_c1_b2_mmdt",
        "j_c2_b1_mmdt",
        "j_c2_b2_mmdt",
        "j_d2_b1_mmdt",
        "j_d2_b2_mmdt",
        "j_d2_a1_b1_mmdt",
        "j_d2_a1_b2_mmdt",
        "j_m2_b1_mmdt",
        "j_m2_b2_mmdt",
        "j_n2_b1_mmdt",
        "j_n2_b2_mmdt",
        "j_mass_mmdt",
        "j_multiplicity",
    ],
    "Labels": ["j_g", "j_q", "j_w", "j_z", "j_t"],
    "KerasModel": "three_layer_model",
    "KerasModelRetrain": "three_layer_model_constraint",
    "KerasLoss": "categorical_crossentropy",
    "L1Reg": 0.0001,
    "NormalizeInputs": True,
    "InputType": "Dense",
    "ApplyPca": False,
    "PcaDimensions": 10,
}


def _download_jsc_dataset(path: Path):
    """
    Download the Jet Substructure dataset from CERNBox if it does not exist

    Args:
        path (Path): save path to the dataset
    """
    try:
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        # Download the file
        subprocess.run(
            f"wget https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v/download -O {path.as_posix()}",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error downloading Jet Substructure dataset: {e}")


def _preprocess_jsc_dataset(path: Path, config: dict = JSC_CONFIG):
    """
    Preprocess the Jet Substructure dataset from the h5 file input

    Args:
        path (Path): path to the h5 file
        config (dict): configuration for preprocessing
    """
    feature_labels = config["Inputs"]
    output_labels = config["Labels"]

    # Load the h5 file
    with h5py.File(path, "r") as h5py_file:
        tree_array = h5py_file["t_allpar_new"][()]

    # Filter input file, deduplicate, and convert inputs / outputs to numpy array
    dataset_df = pd.DataFrame(
        tree_array, columns=list(set(feature_labels + output_labels))
    )
    dataset_df = dataset_df.drop_duplicates()

    # Using the same dataset split as: https://github.com/hls-fpga-machine-learning/pytorch-training/blob/master/train/Data_loader.py
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        dataset_df[feature_labels].values,  # X
        dataset_df[output_labels].values,  # y
        test_size=0.2,
        random_state=42,
    )

    if config["NormalizeInputs"]:
        scaler = preprocessing.StandardScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)

    # Apply dimenionality reduction to the inputs, if specified
    # Convert X from numpy arrays to torch tensors
    if config["ApplyPca"]:
        # Apply dimenionality reduction to the inputs
        with torch.no_grad():
            dim = config["PcaDimensions"]
            X_train_val_fp64 = torch.from_numpy(X_train_val).double()
            X_test_fp64 = torch.from_numpy(X_test).double()
            _, S, V = torch.svd(X_train_val_fp64)
            X_train_val_pca_fp64 = torch.mm(X_train_val_fp64, V[:, 0:dim])
            X_test_pca_fp64 = torch.mm(X_test_fp64, V[:, 0:dim])
            variance_retained = 100 * (S[0:dim].sum() / S.sum())
            logger.debug(f"Dimensions used for PCA: {dim}")
            logger.debug(f"Variance retained (%): {variance_retained}")
            X_train_val = X_train_val_pca_fp64.float()
            X_test = X_test_pca_fp64.float()
    else:
        X_train_val = torch.from_numpy(X_train_val)
        X_test = torch.from_numpy(X_test)

    # Convert y from numpy arrays to torch tensors
    Y_train_val = torch.from_numpy(Y_train_val).float()
    # Output labels are onehot encoded; this converts labels to be index encoded
    Y_train_val = torch.max(Y_train_val.detach(), 1)[1]

    Y_test = torch.from_numpy(Y_test).float()
    Y_test = torch.max(Y_test.detach(), 1)[1]

    torch.save(X_train_val, path.parent / "X_train_val.pt")
    torch.save(Y_train_val, path.parent / "Y_train_val.pt")
    torch.save(X_test, path.parent / "X_test.pt")
    torch.save(Y_test, path.parent / "Y_test.pt")


# Based off example from: https://github.com/hls-fpga-machine-learning/pytorch-training/blob/master/train/Data_loader.py
# Creates a PyTorch Dataset from the h5 file input.
# Returns labels as a one-hot encoded vector.
# Input / output labels are contained in self.feature_labels / self.output_labels respectively
@add_dataset_info(
    name="jsc",
    dataset_source="manual",
    available_splits=("train", "validation", "test"),
    physical_data_point_classification=True,
    num_classes=5,
    num_features=16,
)
class JetSubstructureDataset(Dataset):
    def __init__(self, h5py_file_path: Path, split="train", jsc_config=JSC_CONFIG):
        super().__init__()

        self.split = split
        self.h5py_file_path = h5py_file_path
        self.config = jsc_config

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def prepare_data(self) -> None:
        # Download and preprocess the dataset on the main process in distributed training
        _download_jsc_dataset(self.h5py_file_path)
        _preprocess_jsc_dataset(self.h5py_file_path, self.config)

    def setup(self) -> None:
        # Load the preprocessed dataset on each process in distributed training
        if self.split in ["train", "validation"]:
            x_path = self.h5py_file_path.parent / "X_train_val.pt"
            y_path = self.h5py_file_path.parent / "Y_train_val.pt"
        elif self.split == "test":
            x_path = self.h5py_file_path.parent / "X_test.pt"
            y_path = self.h5py_file_path.parent / "Y_test.pt"
        elif self.split == "pred":
            x_path = self.h5py_file_path.parent / "X_test.pt"
            y_path = self.h5py_file_path.parent / "Y_test.pt"
        else:
            raise ValueError(f"Split {self.split} is not supported for JSC dataset")

        assert (
            x_path.exists() and y_path.exists()
        ), "Dataset not downloaded or preprocessed"

        self.X = torch.load(x_path)
        self.Y = torch.load(y_path)
