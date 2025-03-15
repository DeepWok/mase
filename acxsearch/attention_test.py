import sys
import time
from pathlib import Path
import torch
import argparse
from datetime import datetime

from chop.passes.graph.analysis import add_software_metadata_analysis_pass
from chop.passes.graph.transforms.quantize import (
    quantize_transform_pass,
)
from utils import acc_cal, loss_cal, initialize_graph
import json


ini_args = {
"model_name": "deit_base_patch16_224",
"dataset_name": "imagenet",
"batch_size": 32,
"load_name": None,
"load_type": None,
}
device='cuda:0'
mg, info = initialize_graph(**ini_args)
# for n in mg.fx_graph.nodes:
#     print(n.name)
acc = acc_cal(mg.model, info['data_module'].test_dataloader())
# a = 1

