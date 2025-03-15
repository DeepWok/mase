# TODO:
# check the transformer ViT result
import sys
import time
from pathlib import Path
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
ROOT = "/mnt/data/cx922/past/mase-tools"
sys.path.append(ROOT)
sys.path.append(ROOT + "/machop")
from machop.chop.dataset import get_dataset_info, MaseDataModule
from utils import acc_cal, initialize_graph


ini_args = {
    "model_name": "vit_base_patch16",
    "dataset_name": "cifar10",
    "batch_size": 16,
    "load_name": None,
    "load_type": None,
}
mg, info = initialize_graph(**ini_args)

test_dataloader = info["data_module"].test_dataloader()
acc_after = acc_cal(mg.model, test_dataloader)